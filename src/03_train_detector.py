# Databricks notebook source
# MAGIC %md
# MAGIC # Task 3 · train_vision_detector
# MAGIC Fine-tune an Ultralytics YOLO detector on the synthetic dataset, track in MLflow, register the model in
# MAGIC Unity Catalog with the `champion` alias, and run a sim→real check on the original photo.

# COMMAND ----------
# Ultralytics pulls opencv-python 5.x, whose import aborts the Python process on serverless GPU.
# Swap in headless OpenCV 4.11 before anything imports cv2.
import subprocess, sys
def _sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).returncode
_sh(f"{sys.executable} -m pip uninstall -y -q opencv-python")
assert _sh(f"{sys.executable} -m pip install -q --force-reinstall --no-deps 'opencv-python-headless==4.11.0.86'") == 0
dbutils.library.restartPython()

# COMMAND ----------
for k, d in [("catalog", "main"), ("schema", "synthgen_3d"), ("job_run_id", "0"),
             ("target_class_name", "part"), ("epochs", "30"), ("image_size", "640"), ("base_weights", "yolo26n.pt"),
             ("model_name", "synthgen_detector"), ("experiment_path", "/Shared/synthgen-3d"), ("image_volume_path", "")]:
    dbutils.widgets.text(k, d)
W = {k: dbutils.widgets.get(k) for k in ["catalog", "schema", "job_run_id", "target_class_name", "epochs", "image_size",
                                         "base_weights", "model_name", "experiment_path", "image_volume_path"]}
CAT, SCH, RUN = W["catalog"], W["schema"], int(W["job_run_id"])
VOL = f"/Volumes/{CAT}/{SCH}"
RUNS_TABLE, EPOCHS_TABLE = f"{CAT}.{SCH}.pipeline_runs", f"{CAT}.{SCH}.training_epochs"
MODEL = f"{CAT}.{SCH}.{W['model_name']}"

# COMMAND ----------
import os, sys, json, time, shutil, base64, io, importlib.metadata as md
sys.path.append(os.getcwd())
from synthgen_common import Progress, run_dir, update_run
T0 = time.time()
EPOCHS = int(W["epochs"])
OUT = run_dir(VOL, "synthetic_dataset", RUN)
prog = Progress(OUT, "train")
prog.update(0, EPOCHS, "Preparing dataset")
update_run(spark, RUNS_TABLE, RUN, status="running_train")

try:
    ds_vol = dbutils.jobs.taskValues.get(taskKey="render_synthetic_dataset", key="dataset_path", debugValue="")
except Exception:
    ds_vol = ""
ds_vol = ds_vol or OUT
local = "/tmp/ds"
shutil.rmtree(local, ignore_errors=True)
shutil.copytree(f"{ds_vol}/images", f"{local}/images"); shutil.copytree(f"{ds_vol}/labels", f"{local}/labels")
with open(f"{local}/data.yaml", "w") as f:
    f.write(f"path: {local}\ntrain: images/train\nval: images/val\nnames:\n  0: {W['target_class_name']}\n")

# COMMAND ----------
import torch, mlflow
from mlflow import MlflowClient
from ultralytics import YOLO, settings
mlflow.set_tracking_uri("databricks"); mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(W["experiment_path"])
os.environ.update(MLFLOW_TRACKING_URI="databricks", MLFLOW_EXPERIMENT_NAME=W["experiment_path"],
                  MLFLOW_RUN=f"synthgen-{RUN}", MLFLOW_KEEP_RUN_ACTIVE="True")
settings.update({"mlflow": True})

vram = torch.cuda.get_device_properties(0).total_memory / 1e9
batch = 32 if vram > 60 else 16 if vram > 20 else 8
model = YOLO(W["base_weights"])
epoch_rows = []
def on_epoch_end(trainer):
    m = trainer.metrics or {}
    loss = trainer.label_loss_items(trainer.tloss, prefix="train") if hasattr(trainer, "label_loss_items") else {}
    row = dict(epoch=int(trainer.epoch) + 1, map50=float(m.get("metrics/mAP50(B)", 0)), map50_95=float(m.get("metrics/mAP50-95(B)", 0)),
               precision=float(m.get("metrics/precision(B)", 0)), recall=float(m.get("metrics/recall(B)", 0)),
               box_loss=float(loss.get("train/box_loss", 0) or 0), cls_loss=float(loss.get("train/cls_loss", 0) or 0))
    epoch_rows.append(row)
    prog.update(row["epoch"], EPOCHS, f"Epoch {row['epoch']}/{EPOCHS}", history=epoch_rows)
model.add_callback("on_fit_epoch_end", on_epoch_end)

prog.update(0, EPOCHS, "Training")
t = time.time()
res = model.train(data=f"{local}/data.yaml", epochs=EPOCHS, imgsz=int(W["image_size"]), batch=batch, workers=4,
                  project="/tmp/yolo", name=str(RUN), exist_ok=True, plots=True, verbose=False, seed=0)
train_sec = round(time.time() - t, 1)
final = {k: float(v) for k, v in (getattr(res, "results_dict", None) or {}).items()}
best = f"/tmp/yolo/{RUN}/weights/best.pt"
run = mlflow.active_run() or mlflow.last_active_run()
mlflow_run_id = run.info.run_id

# COMMAND ----------
# Register a pyfunc wrapper in UC (base64 image in -> boxes out) and point @champion at it.
import pandas as pd
from PIL import Image
from mlflow.models import infer_signature

class SynthGenDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from ultralytics import YOLO
        self.m = YOLO(context.artifacts["weights"])
    def predict(self, context, model_input, params=None):
        import base64, io, json, pandas as pd
        from PIL import Image
        conf = float((params or {}).get("conf", 0.25))
        rows = []
        for b64 in model_input["image_b64"]:
            im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            r = self.m.predict(im, conf=conf, verbose=False)[0]
            rows.append({"boxes_xyxy": json.dumps(r.boxes.xyxy.tolist()), "conf": json.dumps(r.boxes.conf.tolist()),
                         "cls": json.dumps(r.boxes.cls.tolist()), "names": json.dumps(r.names)})
        return pd.DataFrame(rows)

prog.update(EPOCHS, EPOCHS, "Registering model in Unity Catalog")
sample_img = sorted(os.listdir(f"{local}/images/val"))[0]
sample = pd.DataFrame({"image_b64": [base64.b64encode(open(f"{local}/images/val/{sample_img}", "rb").read()).decode()]})
sig = infer_signature(sample, pd.DataFrame([{"boxes_xyxy": "[]", "conf": "[]", "cls": "[]", "names": "{}"}]), params={"conf": 0.25})
reqs = [f"ultralytics=={md.version('ultralytics')}", f"torch=={torch.__version__.split('+')[0]}",
        "opencv-python-headless==4.11.0.86", f"mlflow=={md.version('mlflow')}", "pillow", "pandas"]
# Ultralytics' MLflow callback leaves its run active (MLFLOW_KEEP_RUN_ACTIVE=True). Reuse that run
# instead of calling start_run(run_id=...) on top of it, which raises "already active".
# Only start a run when none is active; end it exactly once in finally.
_active = mlflow.active_run()
if _active is not None and _active.info.run_id != mlflow_run_id:
    mlflow.end_run()
    _active = None
if _active is None:
    mlflow.start_run(run_id=mlflow_run_id)
try:
    mlflow.set_tags({"synthgen.job_run_id": str(RUN), "synthgen.source_image": W["image_volume_path"],
                     "synthgen.class": W["target_class_name"]})
    try:
        mlflow.log_input(mlflow.data.load_delta(table_name=f"{CAT}.{SCH}.synthetic_views"), context="training")
    except Exception as e:
        print("dataset lineage not logged:", e)
    for p in ("results.png", "confusion_matrix.png", "val_batch0_pred.jpg", "BoxPR_curve.png"):
        if os.path.exists(f"/tmp/yolo/{RUN}/{p}"):
            mlflow.log_artifact(f"/tmp/yolo/{RUN}/{p}", artifact_path="plots")
    info = mlflow.pyfunc.log_model(name="detector", python_model=SynthGenDetector(), artifacts={"weights": best},
                                   signature=sig, input_example=sample, pip_requirements=reqs,
                                   registered_model_name=MODEL)
    version = str(info.registered_model_version)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL, "champion", version)
    client.update_model_version(MODEL, version, description=f"SynthGen run {RUN}: {W['target_class_name']} detector trained on synthetic IDOL/gsplat renders")
finally:
    mlflow.end_run()

# COMMAND ----------
# Sim -> real: run the detector on the original photo and save an annotated copy for the app.
real_path, real_conf = None, None
if W["image_volume_path"]:
    try:
        r = YOLO(best).predict(W["image_volume_path"], conf=0.1, verbose=False)[0]
        real_conf = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
        Image.fromarray(r.plot()[..., ::-1]).save("/tmp/real_test.jpg", quality=92)
        real_path = f"{OUT}/real_test.jpg"; shutil.copyfile("/tmp/real_test.jpg", real_path)
    except Exception as e:
        print("real test skipped:", e)
for p in ("results.png", "confusion_matrix.png", "val_batch0_pred.jpg"):
    if os.path.exists(f"/tmp/yolo/{RUN}/{p}"):
        os.makedirs(f"{OUT}/plots", exist_ok=True); shutil.copyfile(f"/tmp/yolo/{RUN}/{p}", f"{OUT}/plots/{p}")

# COMMAND ----------
from pyspark.sql import functions as Fs
spark.sql(f"DELETE FROM {EPOCHS_TABLE} WHERE job_run_id = {RUN}")
if epoch_rows:
    spark.createDataFrame([{**r, "job_run_id": RUN} for r in epoch_rows]).withColumn("created_at", Fs.current_timestamp()) \
        .select("job_run_id", "epoch", "map50", "map50_95", "precision", "recall", "box_loss", "cls_loss", "created_at") \
        .write.mode("append").saveAsTable(EPOCHS_TABLE)
update_run(spark, RUNS_TABLE, RUN, status="model_registered", mlflow_run_id=mlflow_run_id, model_name=MODEL,
           model_version=version, map50=final.get("metrics/mAP50(B)"), map50_95=final.get("metrics/mAP50-95(B)"),
           precision=final.get("metrics/precision(B)"), recall=final.get("metrics/recall(B)"),
           train_seconds=float(train_sec), real_test_image_path=real_path, real_test_max_conf=real_conf)
summary = {"model": f"{MODEL}@champion", "version": version, "mlflow_run_id": mlflow_run_id, "train_sec": train_sec,
           "batch": batch, "metrics": final, "real_test_max_conf": real_conf, "total_sec": round(time.time() - T0, 1)}
prog.update(EPOCHS, EPOCHS, "Model registered", **summary)
print(summary)
dbutils.notebook.exit(json.dumps(summary, default=str))
