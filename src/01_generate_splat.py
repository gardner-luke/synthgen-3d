# Databricks notebook source
# MAGIC %md
# MAGIC # Task 1 · generate_3d_splat
# MAGIC Single photo → IDOL → 3D Gaussian splat (.ply) + predicted camera, written to the `splat_models` volume.
# MAGIC Weights come from the staged HF cache tarball in `model_cache` (offline, no Hugging Face calls at run time).

# COMMAND ----------
for k, d in [("catalog", "main"), ("schema", "synthgen_3d"), ("job_run_id", "0"),
             ("image_volume_path", ""), ("target_class_name", "part"), ("gaussian_count", "262144"),
             ("idol_steps", "50"), ("idol_mode", "real"), ("seed", "42"), ("submitted_by", ""),
             ("hf_secret_scope", "synthgen"), ("hf_secret_key", "huggingface"), ("params_json", "{}")]:
    dbutils.widgets.text(k, d)
W = {k: dbutils.widgets.get(k) for k in ["catalog", "schema", "job_run_id", "image_volume_path", "target_class_name",
                                         "gaussian_count", "idol_steps", "idol_mode", "seed", "submitted_by", "params_json"]}
CAT, SCH, RUN = W["catalog"], W["schema"], int(W["job_run_id"])
VOL = f"/Volumes/{CAT}/{SCH}"
RUNS_TABLE = f"{CAT}.{SCH}.pipeline_runs"
assert W["image_volume_path"].startswith("/Volumes/"), "image_volume_path must be a UC Volume path"

# COMMAND ----------
import os, sys, json, time, inspect, shutil
sys.path.append(os.getcwd())
from synthgen_common import cuda_env, untar_to, sha256_file, Progress, run_dir, update_run
cuda_env()
T0 = time.time()
OUT = run_dir(VOL, "splat_models", RUN)
os.makedirs(OUT, exist_ok=True)
prog = Progress(OUT, "generate")
prog.update(0, 4, "Preparing model weights")

src_sha = sha256_file(W["image_volume_path"])
update_run(spark, RUNS_TABLE, RUN, status="running_generate", submitted_by=W["submitted_by"] or None,
           source_image_path=W["image_volume_path"], source_sha256=src_sha,
           target_class_name=W["target_class_name"],
           params=W["params_json"] if W["params_json"] not in ("", "{}") else json.dumps(
               {k: W[k] for k in ("gaussian_count", "idol_steps", "idol_mode", "seed", "target_class_name")}),
           gaussian_count=int(W["gaussian_count"]))
spark.sql(f"UPDATE {RUNS_TABLE} SET submitted_at = coalesce(submitted_at, current_timestamp()) WHERE job_run_id = {RUN}")

# Offline HF cache from the volume; fall back to an online download with the secret if it's missing.
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ.setdefault("ATTN_BACKEND", "sdpa")
t = time.time()
if untar_to(f"{VOL}/model_cache/hf_hub.tar", "/tmp/hf_home/hub"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    weights_source = "volume_cache"
else:
    os.environ["HF_TOKEN"] = dbutils.secrets.get(dbutils.widgets.get("hf_secret_scope"), dbutils.widgets.get("hf_secret_key"))
    weights_source = "huggingface"
restore_sec = round(time.time() - t, 1)
print(f"weights from {weights_source} in {restore_sec}s")

# COMMAND ----------
import torch
from idol import IdolPipeline
prog.update(1, 4, "Loading IDOL")
t = time.time()
pipe = IdolPipeline.from_pretrained("speridlabs/idol")
load_sec = round(time.time() - t, 1)

prog.update(2, 4, "Reconstructing 3D Gaussians")
kwargs = dict(mode=W["idol_mode"], seed=int(W["seed"]), num_gaussians=int(W["gaussian_count"]))
sig = inspect.signature(pipe.__call__).parameters
for name in ("steps", "num_steps", "num_inference_steps"):
    if name in sig:
        kwargs[name] = int(W["idol_steps"]); break
torch.cuda.synchronize(); t = time.time()
out = pipe(W["image_volume_path"], **kwargs)
torch.cuda.synchronize()
infer_sec = round(time.time() - t, 2)

# COMMAND ----------
prog.update(3, 4, "Saving splat")
local = "/tmp/idol_out"; os.makedirs(local, exist_ok=True)
ply_local = f"{local}/model.ply"
out.gaussian.save_ply(ply_local)
camera = out.camera if isinstance(out.camera, dict) else {}
camera_json = json.dumps(camera, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
with open(f"{local}/camera.json", "w") as f:
    f.write(camera_json)
pre = getattr(out, "preprocessed_image", None)
try:
    from PIL import Image
    import numpy as np
    if pre is not None:
        if torch.is_tensor(pre):
            a = pre.detach().float().cpu().numpy()
            a = np.moveaxis(a, 0, -1) if a.ndim == 3 and a.shape[0] in (3, 4) else a
            pre = Image.fromarray((np.clip(a, 0, 1) * 255).astype("uint8") if a.max() <= 1.0 else a.astype("uint8"))
        elif not hasattr(pre, "save"):
            pre = Image.fromarray(np.asarray(pre))
        pre.save(f"{local}/preprocessed.png")
except Exception as e:
    print("preprocessed image not saved:", e)
for f in os.listdir(local):
    shutil.copy(os.path.join(local, f), OUT)

ply_path = f"{OUT}/model.ply"
update_run(spark, RUNS_TABLE, RUN, status="splat_ready", ply_path=ply_path, camera_json=camera_json,
           idol_seconds=float(infer_sec))
dbutils.jobs.taskValues.set("ply_path", ply_path)
dbutils.jobs.taskValues.set("source_sha256", src_sha)
summary = {"weights_source": weights_source, "restore_sec": restore_sec, "load_sec": load_sec, "infer_sec": infer_sec,
           "ply_mb": round(os.path.getsize(ply_local) / 1e6, 1), "device": torch.cuda.get_device_name(0),
           "total_sec": round(time.time() - T0, 1)}
prog.update(4, 4, "Splat ready", **summary)
print(summary)
dbutils.notebook.exit(json.dumps(summary))
