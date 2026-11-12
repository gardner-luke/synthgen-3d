# Databricks notebook source
# MAGIC %md
# MAGIC # SynthGen 3D — one-time setup
# MAGIC Run once per workspace (and again only when you bump IDOL / torch / gsplat versions).
# MAGIC 1. Unity Catalog objects (schema, volumes, Delta tables)
# MAGIC 2. Build the IDOL SDK wheel into `model_cache/wheels/` (so pipeline envs install offline from the Volume)
# MAGIC 3. Download IDOL + gated encoders (DINOv3, FLUX.2 VAE, BiRefNet) once and stage the HF cache as a tarball
# MAGIC 4. Compile gsplat's CUDA extension once (sm_86 + sm_90) and stage the build cache as a tarball
# MAGIC
# MAGIC After this, pipeline runs never touch Hugging Face or recompile CUDA.

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "synthgen_3d")
dbutils.widgets.text("hf_secret_scope", "synthgen")
dbutils.widgets.text("hf_secret_key", "huggingface")
dbutils.widgets.text("sample_image", "/Volumes/main/synthgen_3d/raw_inputs/samples/bearing_fag_6304.jpg")
CAT, SCH = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

# COMMAND ----------
# 1. UC objects
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CAT}.{SCH} COMMENT '3D synthetic data generation demo: IDOL splats -> gsplat renders -> YOLO'")
for v, c in [("raw_inputs", "User-uploaded source photos of parts"),
             ("splat_models", "IDOL 3D Gaussian splats (.ply) + camera JSON per pipeline run"),
             ("synthetic_dataset", "Rendered synthetic views + YOLO labels per pipeline run"),
             ("model_cache", "Staged weights, wheels and CUDA build caches (no runtime downloads)"),
             ("backgrounds", "Optional background images used for domain randomization")]:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CAT}.{SCH}.{v} COMMENT '{c}'")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CAT}.{SCH}.pipeline_runs (
  job_run_id BIGINT COMMENT 'Lakeflow job run id (primary key)',
  submitted_by STRING, submitted_at TIMESTAMP,
  source_image_path STRING, source_sha256 STRING, target_class_name STRING,
  params STRING COMMENT 'JSON of job parameters',
  status STRING COMMENT 'submitted | splat_ready | dataset_ready | model_registered | failed',
  ply_path STRING, camera_json STRING, gaussian_count INT, idol_seconds DOUBLE,
  num_views INT, dataset_path STRING, render_seconds DOUBLE,
  mlflow_run_id STRING, model_name STRING, model_version STRING,
  map50 DOUBLE, map50_95 DOUBLE, precision DOUBLE, recall DOUBLE, train_seconds DOUBLE,
  real_test_image_path STRING, real_test_max_conf DOUBLE,
  updated_at TIMESTAMP
) CLUSTER BY (job_run_id)
COMMENT 'One row per SynthGen pipeline run: provenance from source photo to registered model'
""")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CAT}.{SCH}.synthetic_views (
  job_run_id BIGINT, view_id INT, split STRING,
  image_path STRING, label_path STRING, class_name STRING,
  azimuth_deg DOUBLE, elevation_deg DOUBLE, radius DOUBLE, fov_deg DOUBLE,
  world_to_camera ARRAY<DOUBLE> COMMENT '4x4 row-major, OpenCV convention',
  bbox_xyxy ARRAY<INT> COMMENT 'pixel coords from rendered alpha mask',
  bbox_yolo ARRAY<DOUBLE> COMMENT 'normalized cx, cy, w, h',
  mask_coverage DOUBLE, background_kind STRING, width INT, height INT,
  source_sha256 STRING, created_at TIMESTAMP
) CLUSTER BY (job_run_id)
COMMENT 'Per-image metadata for synthetic renders: camera pose, bbox, lineage to the source photo'
""")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CAT}.{SCH}.training_epochs (
  job_run_id BIGINT, epoch INT, map50 DOUBLE, map50_95 DOUBLE, precision DOUBLE, recall DOUBLE,
  box_loss DOUBLE, cls_loss DOUBLE, created_at TIMESTAMP
) CLUSTER BY (job_run_id)
COMMENT 'Per-epoch validation metrics from detector training (mirrors MLflow, cheap for the app to read)'
""")
print("UC objects ready")

# COMMAND ----------
# 2. Build IDOL wheel (pip upgrade first: old pip + hatchling fails on packaging.licenses)
import os, sys, json, time, shutil, glob
sys.path.append(os.getcwd())
from synthgen_common import sh, pip, cuda_env, tar_dir
VOL = f"/Volumes/{CAT}/{SCH}"
WHEELS = f"{VOL}/model_cache/wheels"
os.makedirs(WHEELS, exist_ok=True)
REPORT = {}
rc, out = pip("-U pip 'packaging>=24.2' hatchling wheel")
rc, commit = sh("git ls-remote https://github.com/speridlabs/idol-sdk.git HEAD | cut -f1")
REPORT["idol_commit"] = commit.strip().splitlines()[0] if commit.strip() else None
rc, out = sh(f"{sys.executable} -m pip wheel --no-deps -w /tmp/wheels 'git+https://github.com/speridlabs/idol-sdk.git@{REPORT['idol_commit']}'")
assert rc == 0, out
for w in glob.glob("/tmp/wheels/idol*.whl"):
    shutil.copy(w, WHEELS)
REPORT["idol_wheels"] = os.listdir(WHEELS)
rc, out = pip(" ".join(glob.glob("/tmp/wheels/idol*.whl")))
assert rc == 0, out
json.dump(REPORT, open("/tmp/setup_report.json", "w"))
print(REPORT)

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os, sys, json, time, shutil, glob, importlib.metadata as md
sys.path.append(os.getcwd())
from synthgen_common import cuda_env, tar_dir
CAT, SCH = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
VOL = f"/Volumes/{CAT}/{SCH}"
REPORT = json.load(open("/tmp/setup_report.json"))
cuda_env()
import torch
TAG = f"torch{torch.__version__.split('+')[0]}_cu{torch.version.cuda.replace('.', '')}"
REPORT.update(torch=torch.__version__, tag=TAG, device=torch.cuda.get_device_name(0))

# 3. Download weights once via a real inference (pulls every lazily-loaded component), then stage the hub cache.
os.environ["HF_TOKEN"] = dbutils.secrets.get(dbutils.widgets.get("hf_secret_scope"), dbutils.widgets.get("hf_secret_key"))
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ.setdefault("ATTN_BACKEND", "sdpa")
from idol import IdolPipeline
t = time.time()
pipe = IdolPipeline.from_pretrained("speridlabs/idol")
out = pipe(dbutils.widgets.get("sample_image"), mode="real", seed=42, num_gaussians=262144)
os.makedirs("/tmp/setup_out", exist_ok=True)
out.gaussian.save_ply("/tmp/setup_out/sample.ply")
REPORT["idol_first_run_sec"] = round(time.time() - t, 1)
del pipe; torch.cuda.empty_cache()

hub = "/tmp/hf_home/hub"
REPORT["hub_gb"] = round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(hub) for f in fs
                             if not os.path.islink(os.path.join(d, f))) / 1e9, 2)
t = time.time()
tar_dir(hub, f"{VOL}/model_cache/hf_hub.tar")
REPORT["hub_tar_sec"] = round(time.time() - t, 1)
print(REPORT)

# COMMAND ----------
# 4. Compile gsplat once and stage the torch extension build directory.
os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_ext"
t = time.time()
import numpy as np
from gsplat import rasterization
n = 1000
dev = "cuda"
means = torch.randn(n, 3, device=dev) * 0.2
quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)
scales = torch.full((n, 3), 0.01, device=dev)
opac = torch.full((n,), 0.9, device=dev)
colors = torch.rand(n, 1, 3, device=dev)
vm = torch.eye(4, device=dev)[None]; vm[0, 2, 3] = 2.0
K = torch.tensor([[300.0, 0, 160], [0, 300.0, 160], [0, 0, 1]], device=dev)[None]
rgb, alpha, _ = rasterization(means, quats, scales, opac, colors, vm, K, 320, 320, sh_degree=0)
torch.cuda.synchronize()
REPORT["gsplat_compile_sec"] = round(time.time() - t, 1)
REPORT["torch_ext_contents"] = os.listdir("/tmp/torch_ext") if os.path.exists("/tmp/torch_ext") else None
assert REPORT["torch_ext_contents"], "gsplat did not build into TORCH_EXTENSIONS_DIR"
tar_dir("/tmp/torch_ext", f"{VOL}/model_cache/torch_ext/{TAG}.tar")
REPORT["gsplat"] = md.version("gsplat")
json.dump(REPORT, open(f"{VOL}/model_cache/setup_report.json", "w"), indent=2)
print(json.dumps(REPORT, indent=2))
dbutils.notebook.exit(json.dumps(REPORT))
