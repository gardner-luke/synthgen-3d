# Databricks notebook source
# MAGIC %md
# MAGIC # SynthGen 3D — spike 2: YOLO + MLflow + UC, and a precompiled gsplat wheel
# MAGIC - ultralytics with opencv-python-headless (spike 1 aborted importing GUI cv2)
# MAGIC - full mlflow (not skinny) for pyfunc logging + UC registration
# MAGIC - build a gsplat wheel from source for sm_86 (A10G) + sm_90 (H100) into the model_cache volume

# COMMAND ----------
import json, os, sys, time, shutil, subprocess, traceback, importlib.metadata as md
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]; os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
VOL = "/Volumes/main/synthgen_3d"
RES, VOLRES = "/tmp/spike2.json", f"{VOL}/synthetic_dataset/spike/results_2.json"
os.makedirs(os.path.dirname(VOLRES), exist_ok=True)
R = {"checks": {}}
def save():
    json.dump(R, open(RES, "w"), indent=2, default=str); shutil.copy(RES, VOLRES)
def sh(cmd, timeout=3000, env=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout + "\n" + p.stderr)[-2500:]
def ver(p):
    try: return md.version(p)
    except Exception: return None

t = time.time(); c = {}
rc, out = sh(f"{sys.executable} -m pip install -q -U pip ultralytics 'mlflow>=3.1' plyfile ninja wheel setuptools")
c["rc"], c["tail"] = rc, out[-800:]
rc, out = sh(f"{sys.executable} -m pip uninstall -y opencv-python opencv-contrib-python")
c["uninstall_cv_tail"] = out[-300:]
rc, out = sh(f"{sys.executable} -m pip install -q --force-reinstall --no-deps 'opencv-python-headless==4.11.0.86'")
c["headless_rc"] = rc
c["sec"] = round(time.time() - t, 1)
R["checks"]["install"] = c; save(); print(c)

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import json, os, sys, time, shutil, subprocess, traceback, importlib.metadata as md
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]; os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
CAT, SCH = "main", "synthgen_3d"; VOL = f"/Volumes/{CAT}/{SCH}"
RES, VOLRES = "/tmp/spike2.json", f"{VOL}/synthetic_dataset/spike/results_2.json"
R = json.load(open(RES))
def save():
    json.dump(R, open(RES, "w"), indent=2, default=str); shutil.copy(RES, VOLRES)
def sh(cmd, timeout=3000, env=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout + "\n" + p.stderr)[-2500:]
def ver(p):
    try: return md.version(p)
    except Exception: return None
import torch
R["checks"]["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "ultralytics": ver("ultralytics"),
                      "mlflow": ver("mlflow"), "cv2_headless": ver("opencv-python-headless"), "cv2_gui": ver("opencv-python")}
save(); print(R["checks"]["env"])

# COMMAND ----------
# Build a gsplat wheel once (sm_86 A10G + sm_90 H100) and stash it in the volume.
c = {}
try:
    env = os.environ.copy(); env["TORCH_CUDA_ARCH_LIST"] = "8.6;9.0"; env["MAX_JOBS"] = str(os.cpu_count() or 8)
    tag = f"torch{torch.__version__.split('+')[0]}_cu{torch.version.cuda.replace('.', '')}"
    out_dir = f"/tmp/wheels"; os.makedirs(out_dir, exist_ok=True)
    t = time.time()
    rc, out = sh(f"{sys.executable} -m pip wheel --no-deps --no-build-isolation -w {out_dir} 'git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3'", env=env, timeout=3600)
    c["rc"], c["sec"], c["tail"] = rc, round(time.time() - t, 1), out[-1200:]
    whls = [f for f in os.listdir(out_dir) if f.endswith(".whl")]
    c["wheels"] = whls
    if whls:
        dst = f"{VOL}/model_cache/wheels/{tag}"; os.makedirs(dst, exist_ok=True)
        for f in whls: shutil.copy(os.path.join(out_dir, f), dst)
        c["volume_path"] = dst
        c["wheel_mb"] = round(os.path.getsize(os.path.join(out_dir, whls[0])) / 1e6, 1)
    c["cpu_count"] = os.cpu_count()
    c["ok"] = rc == 0 and bool(whls)
except Exception:
    c["ok"] = False; c["error"] = traceback.format_exc()[-2000:]
R["checks"]["gsplat_wheel"] = c; save(); print(json.dumps(c, indent=2)[:3000])

# COMMAND ----------
# Probe cv2 + ultralytics import in a subprocess so an abort can't kill the notebook.
c = {}
for mod in ("cv2", "ultralytics"):
    rc, out = sh(f"{sys.executable} -X faulthandler -c \"import {mod}; print('OK', getattr({mod}, '__version__', ''))\"", timeout=300)
    c[mod] = {"rc": rc, "tail": out[-1500:]}
R["checks"]["import_probe"] = c; save(); print(json.dumps(c, indent=1)[:3000])
IMPORT_OK = all(v["rc"] == 0 for v in c.values())

# COMMAND ----------
# YOLO tiny train -> MLflow -> pyfunc -> UC
c = {}
try:
    assert IMPORT_OK, 'cv2/ultralytics import aborts in subprocess; see import_probe'
    import cv2; c["cv2"] = cv2.__version__
    import mlflow
    from ultralytics import YOLO, settings
    user = spark.sql("select current_user()").first()[0]
    exp = f"/Users/{user}/synthgen-3d-spike"
    mlflow.set_tracking_uri("databricks"); mlflow.set_registry_uri("databricks-uc"); mlflow.set_experiment(exp)
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"; os.environ["MLFLOW_EXPERIMENT_NAME"] = exp
    settings.update({"mlflow": True})
    m, c["weights"] = None, None
    for w in ("yolo26n.pt", "yolo11n.pt"):
        try: m = YOLO(w); c["weights"] = w; break
        except Exception as e: c[f"load_{w}"] = repr(e)[:300]
    t = time.time()
    res = m.train(data="coco8.yaml", epochs=3, imgsz=320, batch=8, workers=2, project="/tmp/yolo", name="spike", exist_ok=True, verbose=False, plots=False)
    c["train_sec"] = round(time.time() - t, 1)
    c["metrics"] = {k: float(v) for k, v in (getattr(res, "results_dict", None) or {}).items()}
    lr = mlflow.last_active_run(); c["autolog_run_id"] = lr.info.run_id if lr else None
    if lr: c["autolog_metric_keys"] = sorted(mlflow.get_run(lr.info.run_id).data.metrics.keys())[:40]
    best = "/tmp/yolo/spike/weights/best.pt"; c["best_exists"] = os.path.exists(best)

    import pandas as pd, base64, io
    from PIL import Image
    class YoloPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from ultralytics import YOLO
            self.m = YOLO(context.artifacts["weights"])
        def predict(self, context, model_input):
            import base64, io, json, pandas as pd
            from PIL import Image
            rows = []
            for b64 in model_input["image_b64"]:
                im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                r = self.m.predict(im, verbose=False)[0]
                rows.append({"boxes_xyxy": json.dumps(r.boxes.xyxy.tolist()), "conf": json.dumps(r.boxes.conf.tolist()), "cls": json.dumps(r.boxes.cls.tolist())})
            return pd.DataFrame(rows)
    buf = io.BytesIO(); Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, format="PNG")
    sample = pd.DataFrame({"image_b64": [base64.b64encode(buf.getvalue()).decode()]})
    from mlflow.models import infer_signature
    sig = infer_signature(sample, pd.DataFrame([{"boxes_xyxy": "[]", "conf": "[]", "cls": "[]"}]))
    t = time.time()
    with mlflow.start_run(run_name="spike-register"):
        info = mlflow.pyfunc.log_model(name="detector", python_model=YoloPyfunc(), artifacts={"weights": best},
                                       signature=sig, input_example=sample,
                                       pip_requirements=[f"ultralytics=={ver('ultralytics')}", f"mlflow=={ver('mlflow')}", "opencv-python-headless==4.11.0.86", "pillow", "pandas"],
                                       registered_model_name=f"{CAT}.{SCH}.spike_detector")
    c["register_sec"] = round(time.time() - t, 1)
    c["model_version"] = getattr(info, "registered_model_version", None)
    from mlflow import MlflowClient
    MlflowClient().set_registered_model_alias(f"{CAT}.{SCH}.spike_detector", "champion", c["model_version"])
    loaded = mlflow.pyfunc.load_model(f"models:/{CAT}.{SCH}.spike_detector@champion")
    c["reload_predict"] = loaded.predict(sample).to_dict("records")
    c["ok"] = True
except Exception:
    c["ok"] = False; c["error"] = traceback.format_exc()[-3000:]
R["checks"]["yolo_mlflow_uc"] = c; save(); print(json.dumps(c, indent=2, default=str)[:4000])

# COMMAND ----------
dbutils.notebook.exit(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in R["checks"].items()}))
