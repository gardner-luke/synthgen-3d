# Databricks notebook source
# MAGIC %md
# MAGIC # SynthGen 3D — serverless GPU feasibility spike
# MAGIC One GPU allocation, five checks. Each check records pass/fail + timings into a results dict
# MAGIC persisted to /tmp (survives restartPython) and finally to the volume.
# MAGIC 1. Environment probe (python / torch / CUDA / nvcc / disk / Spark)
# MAGIC 2. IDOL install + inference (cold + warm), stage HF cache to `model_cache`
# MAGIC 3. gsplat install + render 8 views + alpha-mask bboxes
# MAGIC 4. Delta write from the GPU task
# MAGIC 5. ultralytics train (tiny) + MLflow + UC registration

# COMMAND ----------
import json, os, sys, time, shutil, subprocess, platform, traceback, importlib.metadata as md

for _d in ("/usr/local/cuda/bin", "/usr/local/cuda-12/bin"):
    if os.path.exists(_d): os.environ["PATH"] = _d + ":" + os.environ["PATH"]
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
T0 = time.time()
R = {"started_epoch": T0, "checks": {}}
RES = "/tmp/spike_results.json"

VOLRES = "/Volumes/main/synthgen_3d/synthetic_dataset/spike/results.json"
os.makedirs(os.path.dirname(VOLRES), exist_ok=True)
def save():
    json.dump(R, open(RES, "w"), indent=2, default=str)
    shutil.copy(RES, VOLRES)

def sh(cmd, timeout=1800):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + "\n" + p.stderr)[-3000:]

def ver(pkg):
    try: return md.version(pkg)
    except Exception: return None

rc, smi = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
rc2, nvcc = sh("nvcc --version || ls /usr/local/cuda*/bin/nvcc")
R["checks"]["1_env_pre"] = {
    "python": sys.version.split()[0], "platform": platform.platform(),
    "nvidia_smi": smi.strip(), "nvcc": nvcc.strip()[-400:], "nvcc_on_path": shutil.which("nvcc"),
    "CUDA_HOME": os.environ.get("CUDA_HOME"),
    "pre_torch": ver("torch"), "pre_torchvision": ver("torchvision"), "pre_mlflow": ver("mlflow"),
    "pre_ultralytics": ver("ultralytics"), "pre_numpy": ver("numpy"),
    "disk_tmp_free_gb": round(shutil.disk_usage("/tmp").free / 1e9, 1),
    "disk_local_disk0_free_gb": round(shutil.disk_usage("/local_disk0").free / 1e9, 1) if os.path.exists("/local_disk0") else None,
}
save(); print(json.dumps(R["checks"]["1_env_pre"], indent=2))

# COMMAND ----------
# Install deps. Keep the AI env's torch/ultralytics/mlflow; add IDOL, plyfile, gsplat.
t = time.time()
inst = {}
rc, out = sh(f"{sys.executable} -m pip install -q -U pip 'packaging>=24.2' hatchling ninja plyfile 'huggingface_hub>=0.25'")
inst["prep_rc"], inst["prep_tail"] = rc, out[-600:]
IDOL_SRC = "git+https://github.com/speridlabs/idol-sdk.git"
rc, out = sh(f"{sys.executable} -m pip install -q '{IDOL_SRC}'")
inst["idol_rc"], inst["idol_tail"] = rc, out[-1200:]
if rc != 0:
    rc, out = sh(f"{sys.executable} -m pip install -q --no-build-isolation '{IDOL_SRC}'")
    inst["idol_nobuildiso_rc"], inst["idol_nobuildiso_tail"] = rc, out[-1500:]
rc, tv = sh(f"{sys.executable} -c \"import torch;print('TV', torch.__version__, torch.version.cuda)\"")
tvl = [l for l in tv.splitlines() if l.startswith("TV ")]
inst["torch_after_idol"] = tvl[-1] if tvl else tv[-300:]
rc, out = sh(f"{sys.executable} -m pip install -q gsplat")
inst["gsplat_rc"], inst["gsplat_tail"] = rc, out[-800:]
for pkg in ("ultralytics", "mlflow", "mlflow-skinny"):
    inst[f"have_{pkg}"] = ver(pkg)
if not ver("ultralytics"):
    rc, out = sh(f"{sys.executable} -m pip install -q ultralytics"); inst["ultra_rc"] = rc
inst["install_sec"] = round(time.time() - t, 1)
R["checks"]["0_install"] = inst
save(); print(json.dumps(inst, indent=2)[:4000])

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import json, os, sys, time, shutil, subprocess, traceback, importlib.metadata as md
RES = "/tmp/spike_results.json"
R = json.load(open(RES))
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]; os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
VOLRES = "/Volumes/main/synthgen_3d/synthetic_dataset/spike/results.json"
def save():
    json.dump(R, open(RES, "w"), indent=2, default=str)
    shutil.copy(RES, VOLRES)
def ver(p):
    try: return md.version(p)
    except Exception: return None

CAT, SCH = "main", "synthgen_3d"
VOL = f"/Volumes/{CAT}/{SCH}"
IMG = f"{VOL}/raw_inputs/samples/bearing_fag_6304.jpg"
OUT_LOCAL = "/tmp/spike_out"; os.makedirs(OUT_LOCAL, exist_ok=True)
OUT_VOL = f"{VOL}/splat_models/spike"; os.makedirs(OUT_VOL, exist_ok=True)

import torch
R["checks"]["1_env_post"] = {
    "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
    "idol_sdk": ver("idol-sdk") or ver("idol"), "gsplat": ver("gsplat"), "ultralytics": ver("ultralytics"), "mlflow": ver("mlflow"),
}
save(); print(R["checks"]["1_env_post"])

# COMMAND ----------
# 2. IDOL inference. HF cache on local disk; token from existing secret.
c = {}
try:
    os.environ["HF_TOKEN"] = dbutils.secrets.get("synthgen", "huggingface")
    os.environ["HF_HOME"] = "/tmp/hf_home"
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    t = time.time()
    from idol import IdolPipeline
    pipe = IdolPipeline.from_pretrained("speridlabs/idol")
    c["load_sec"] = round(time.time() - t, 1)
    for i, n in enumerate([262144, 262144, 65536]):
        torch.cuda.synchronize(); t = time.time()
        out = pipe(IMG, mode="real", seed=42, num_gaussians=n)
        torch.cuda.synchronize()
        c[f"infer_{i}_n{n}_sec"] = round(time.time() - t, 2)
    c["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    c["output_attrs"] = [a for a in dir(out) if not a.startswith("_")]
    ply = f"{OUT_LOCAL}/bearing.ply"
    out = pipe(IMG, mode="real", seed=42, num_gaussians=262144)
    out.gaussian.save_ply(ply)
    c["ply_mb"] = round(os.path.getsize(ply) / 1e6, 1)
    for a in ("camera", "cameras", "input_image", "image"):
        if hasattr(out, a): c[f"has_{a}"] = type(getattr(out, a)).__name__
    del pipe; torch.cuda.empty_cache()
    c["ok"] = True
except Exception as e:
    c["ok"] = False; c["error"] = traceback.format_exc()[-2500:]
    # Fallback: CLI entry point (also writes _camera.json)
    try:
        t = time.time()
        p = subprocess.run([sys.executable, "-m", "idol.sample", "--image", IMG, "--mode", "real", "--out", OUT_LOCAL],
                           capture_output=True, text=True, timeout=1800, env=os.environ.copy())
        c["cli_rc"], c["cli_sec"], c["cli_tail"] = p.returncode, round(time.time() - t, 1), (p.stdout + p.stderr)[-2000:]
    except Exception:
        c["cli_error"] = traceback.format_exc()[-1500:]
c["out_files"] = os.listdir(OUT_LOCAL)
R["checks"]["2_idol"] = c; save(); print(json.dumps(c, indent=2)[:4000])

# COMMAND ----------
# Inspect PLY header (SH degree, gaussian count) and copy outputs + HF cache to volumes.
c = {}
try:
    from plyfile import PlyData
    plys = [f for f in os.listdir(OUT_LOCAL) if f.endswith(".ply")]
    ply = os.path.join(OUT_LOCAL, plys[0])
    v = PlyData.read(ply)["vertex"]
    names = list(v.data.dtype.names)
    n_rest = len([n for n in names if n.startswith("f_rest_")])
    c.update(ply=ply, count=len(v.data), props=names, f_rest=n_rest, sh_degree=int(round(((n_rest // 3) + 1) ** 0.5 - 1)))
    for f in os.listdir(OUT_LOCAL):
        shutil.copy(os.path.join(OUT_LOCAL, f), OUT_VOL)
    t = time.time()
    if os.path.exists("/tmp/hf_home"):
        shutil.copytree("/tmp/hf_home", f"{VOL}/model_cache/hf_home", dirs_exist_ok=True)
    c["hf_cache_copy_sec"] = round(time.time() - t, 1)
    c["hf_cache_gb"] = round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk("/tmp/hf_home") for f in fs) / 1e9, 2) if os.path.exists("/tmp/hf_home") else None
    c["ok"] = True
except Exception:
    c["ok"] = False; c["error"] = traceback.format_exc()[-2000:]
R["checks"]["2b_ply_and_cache"] = c; save(); print(json.dumps({k: v for k, v in c.items() if k != "props"}, indent=2))

# COMMAND ----------
# 3. gsplat render: 8 orbit views, alpha-mask bboxes.
import numpy as np
c = {}
try:
    from plyfile import PlyData
    from PIL import Image, ImageDraw
    v = PlyData.read(R["checks"]["2b_ply_and_cache"]["ply"])["vertex"].data
    dev = "cuda"
    def col(*ns): return torch.tensor(np.stack([v[n] for n in ns], -1), dtype=torch.float32, device=dev)
    means = col("x", "y", "z")
    quats = col("rot_0", "rot_1", "rot_2", "rot_3")
    scales = torch.exp(col("scale_0", "scale_1", "scale_2"))
    opac = torch.sigmoid(torch.tensor(v["opacity"], dtype=torch.float32, device=dev))
    dc = col("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :]
    rest_names = sorted([n for n in v.dtype.names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    if rest_names:
        rest = col(*rest_names).reshape(len(v), 3, -1).transpose(1, 2)
        shs = torch.cat([dc, rest], 1)
    else:
        shs = dc
    deg = int(round(shs.shape[1] ** 0.5 - 1))
    keep = opac > 0.05
    center = means[keep].median(0).values
    ext = (means[keep] - center).norm(dim=1).quantile(0.98).item()
    c.update(sh_degree=deg, n=len(v), center=center.tolist(), extent=ext)

    def look_at(eye, target, up=np.array([0, 1.0, 0])):
        # OpenCV camera: x right, y down, z forward. Returns world->camera 4x4.
        f = target - eye; f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9
        d = np.cross(f, r)
        Rm = np.stack([r, d, f], 0)
        M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = -Rm @ eye
        return M
    W = H = 640; fov = np.deg2rad(40); fx = 0.5 * W / np.tan(fov / 2)
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)
    cen = center.cpu().numpy(); rad = ext / np.tan(fov / 2) * 1.3
    views = []
    for i in range(8):
        az = 2 * np.pi * i / 8; el = np.deg2rad(20 if i % 2 else -10)
        eye = cen + rad * np.array([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)])
        views.append(look_at(eye, cen))
    viewmats = torch.tensor(np.stack(views), dtype=torch.float32, device=dev)
    from gsplat import rasterization
    for attempt in ("first_call_incl_jit", "warm"):
        torch.cuda.synchronize(); t = time.time()
        rgb, alpha, meta = rasterization(means, quats, scales, opac, shs, viewmats, K[None].expand(8, 3, 3),
                                         W, H, sh_degree=deg, packed=True)
        torch.cuda.synchronize(); c[f"render8_{attempt}_sec"] = round(time.time() - t, 3)
    boxes = []
    out_dir = f"{VOL}/synthetic_dataset/spike"; os.makedirs(out_dir, exist_ok=True)
    for i in range(8):
        a = alpha[i, ..., 0].cpu().numpy(); m = a > 0.5
        ys, xs = np.where(m)
        bb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        boxes.append({"view": i, "bbox": bb, "coverage": round(float(m.mean()), 4)})
        img = Image.fromarray((rgb[i].clamp(0, 1).cpu().numpy() * 255).astype("uint8"))
        if bb: ImageDraw.Draw(img).rectangle(bb, outline=(255, 54, 33), width=3)
        img.save(f"/tmp/view_{i}.png"); shutil.copy(f"/tmp/view_{i}.png", out_dir)
    c["boxes"] = boxes; c["ok"] = True
except Exception:
    c["ok"] = False; c["error"] = traceback.format_exc()[-3000:]
R["checks"]["3_gsplat"] = c; save(); print(json.dumps(c, indent=2)[:4000])

# COMMAND ----------
# 4. Delta write from inside the GPU task.
c = {}
try:
    t = time.time()
    df = spark.createDataFrame([(k, json.dumps(v, default=str)[:20000]) for k, v in R["checks"].items()], "check STRING, payload STRING")
    df.write.mode("overwrite").saveAsTable(f"{CAT}.{SCH}.spike_results")
    c["rows"] = spark.table(f"{CAT}.{SCH}.spike_results").count()
    c["sec"] = round(time.time() - t, 1); c["ok"] = True
except Exception:
    c["ok"] = False; c["error"] = traceback.format_exc()[-2000:]
R["checks"]["4_delta"] = c; save(); print(c)

# COMMAND ----------
# 5. ultralytics tiny train + MLflow + UC register.
c = {}
try:
    import mlflow
    from ultralytics import YOLO, settings
    user = spark.sql("select current_user()").first()[0]
    exp = f"/Users/{user}/synthgen-3d-spike"
    mlflow.set_tracking_uri("databricks"); mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(exp)
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"; os.environ["MLFLOW_EXPERIMENT_NAME"] = exp
    settings.update({"mlflow": True})
    weights = None
    for w in ("yolo26n.pt", "yolo11n.pt"):
        try: m = YOLO(w); weights = w; break
        except Exception as e: c[f"load_{w}"] = repr(e)[:300]
    c["weights"] = weights
    t = time.time()
    res = m.train(data="coco8.yaml", epochs=2, imgsz=320, batch=8, workers=2, project="/tmp/yolo", name="spike", exist_ok=True, verbose=False)
    c["train_sec"] = round(time.time() - t, 1)
    c["metrics"] = {k: float(v) for k, v in (res.results_dict or {}).items()}
    run = mlflow.last_active_run()
    c["autolog_run_id"] = run.info.run_id if run else None
    best = "/tmp/yolo/spike/weights/best.pt"

    class YoloPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from ultralytics import YOLO
            self.m = YOLO(context.artifacts["weights"])
        def predict(self, context, model_input):
            import base64, io, pandas as pd
            from PIL import Image
            rows = []
            for b64 in model_input["image_b64"]:
                im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                r = self.m.predict(im, verbose=False)[0]
                rows.append({"boxes_xyxy": json.dumps(r.boxes.xyxy.tolist()), "conf": json.dumps(r.boxes.conf.tolist()), "cls": json.dumps(r.boxes.cls.tolist())})
            return pd.DataFrame(rows)
    import pandas as pd, base64
    from mlflow.models import infer_signature
    sample = pd.DataFrame({"image_b64": [base64.b64encode(open("/tmp/view_0.png", "rb").read()).decode()]})
    sig = infer_signature(sample, pd.DataFrame([{"boxes_xyxy": "[]", "conf": "[]", "cls": "[]"}]))
    t = time.time()
    with mlflow.start_run(run_name="spike-register") as r:
        info = mlflow.pyfunc.log_model(name="detector", python_model=YoloPyfunc(), artifacts={"weights": best},
                                       signature=sig, input_example=sample,
                                       pip_requirements=[f"ultralytics=={ver('ultralytics')}", f"mlflow=={ver('mlflow')}", "pillow", "pandas"],
                                       registered_model_name=f"{CAT}.{SCH}.spike_detector")
    c["register_sec"] = round(time.time() - t, 1)
    c["model_version"] = getattr(info, "registered_model_version", None)
    c["ok"] = True
except Exception:
    c["ok"] = False; c["error"] = traceback.format_exc()[-3000:]
R["checks"]["5_yolo_mlflow_uc"] = c; save(); print(json.dumps(c, indent=2)[:4000])

# COMMAND ----------
R["total_notebook_sec"] = round(time.time() - R["started_epoch"], 1)
save()
os.makedirs(f"{VOL}/synthetic_dataset/spike", exist_ok=True)
shutil.copy(RES, f"{VOL}/synthetic_dataset/spike/results.json")
dbutils.notebook.exit(json.dumps({k: v.get("ok", None) if isinstance(v, dict) else v for k, v in R["checks"].items()} | {"total_sec": R["total_notebook_sec"]}))
