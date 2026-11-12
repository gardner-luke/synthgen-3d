# Databricks notebook source
# MAGIC %md
# MAGIC # Task 2 · render_synthetic_dataset
# MAGIC Orbit cameras around the splat, rasterize with gsplat, derive tight boxes from the rendered alpha mask,
# MAGIC composite onto randomized backgrounds, and write a YOLO dataset + per-image metadata to Delta.

# COMMAND ----------
for k, d in [("catalog", "main"), ("schema", "synthgen_3d"), ("job_run_id", "0"),
             ("target_class_name", "part"), ("num_views", "300"), ("image_size", "640"), ("seed", "42"),
             ("val_fraction", "0.15"), ("elevation_min_deg", "-20"), ("elevation_max_deg", "70")]:
    dbutils.widgets.text(k, d)
W = {k: dbutils.widgets.get(k) for k in ["catalog", "schema", "job_run_id", "target_class_name", "num_views",
                                         "image_size", "seed", "val_fraction", "elevation_min_deg", "elevation_max_deg"]}
CAT, SCH, RUN = W["catalog"], W["schema"], int(W["job_run_id"])
VOL = f"/Volumes/{CAT}/{SCH}"
RUNS_TABLE, VIEWS_TABLE = f"{CAT}.{SCH}.pipeline_runs", f"{CAT}.{SCH}.synthetic_views"
N, S = int(W["num_views"]), int(W["image_size"])

# COMMAND ----------
import os, sys, json, time, math, shutil, random
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.getcwd())
from synthgen_common import cuda_env, untar_to, tar_dir, Progress, run_dir, update_run
cuda_env()
T0 = time.time()
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TAG = f"torch{torch.__version__.split('+')[0]}_cu{torch.version.cuda.replace('.', '')}"
os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_ext"
EXT_TAR = f"{VOL}/model_cache/torch_ext/{TAG}.tar"
ext_restored = untar_to(EXT_TAR, "/tmp/torch_ext")

OUT = run_dir(VOL, "synthetic_dataset", RUN)
prog = Progress(OUT, "render")
prog.update(0, N, "Loading splat")
update_run(spark, RUNS_TABLE, RUN, status="running_render")

try:
    ply_path = dbutils.jobs.taskValues.get(taskKey="generate_3d_splat", key="ply_path", debugValue="")
except Exception:
    ply_path = ""
ply_path = ply_path or f"{run_dir(VOL, 'splat_models', RUN)}/model.ply"
src_sha = spark.sql(f"SELECT source_sha256 FROM {RUNS_TABLE} WHERE job_run_id = {RUN}").first()[0]

# COMMAND ----------
# Load the 3DGS PLY into tensors (standard INRIA layout; IDOL writes SH degree 0).
from plyfile import PlyData
v = PlyData.read(ply_path)["vertex"].data
dev = "cuda"
def col(*names):
    return torch.tensor(np.stack([v[n] for n in names], -1), dtype=torch.float32, device=dev)
means = col("x", "y", "z")
quats = col("rot_0", "rot_1", "rot_2", "rot_3")
scales = torch.exp(col("scale_0", "scale_1", "scale_2"))
opac = torch.sigmoid(torch.tensor(np.asarray(v["opacity"]), dtype=torch.float32, device=dev))
dc = col("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :]
rest = sorted([n for n in v.dtype.names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
shs = torch.cat([dc, col(*rest).reshape(len(v), 3, -1).transpose(1, 2)], 1) if rest else dc
SH_DEG = int(round(math.sqrt(shs.shape[1]) - 1))
solid = opac > 0.05
center = means[solid].median(0).values
extent = (means[solid] - center).norm(dim=1).quantile(0.98).item()
print(f"{len(v)} gaussians, SH degree {SH_DEG}, extent {extent:.3f}")

# COMMAND ----------
# Camera sampling: random azimuth / elevation band / distance / FOV / roll, with target jitter so the part
# isn't always centered. OpenCV convention (x right, y down, z forward), world->camera matrices.
rng = np.random.default_rng(int(W["seed"]) + RUN % 100000)
el_lo, el_hi = float(W["elevation_min_deg"]), float(W["elevation_max_deg"])

def look_at(eye, target, roll):
    f = target - eye; f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0])
    if abs(f @ up) > 0.98:
        up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    c, s = math.cos(roll), math.sin(roll)
    r, d = c * r + s * d, -s * r + c * d
    R = np.stack([r, d, f], 0)
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ eye
    return M

cams = []
cen = center.cpu().numpy()
for i in range(N):
    az = rng.uniform(0, 360); el = rng.uniform(el_lo, el_hi)
    fov = rng.uniform(35, 55)
    base = extent / math.tan(math.radians(fov) / 2)
    radius = base * rng.uniform(1.15, 2.6)
    a, e = math.radians(az), math.radians(el)
    eye = cen + radius * np.array([math.cos(e) * math.sin(a), -math.sin(e), math.cos(e) * math.cos(a)])
    target = cen + rng.uniform(-0.35, 0.35, 3) * extent
    roll = math.radians(rng.uniform(-20, 20))
    fx = 0.5 * S / math.tan(math.radians(fov) / 2)
    cams.append(dict(view_id=i, az=az, el=el, radius=radius, fov=fov, w2c=look_at(eye, target, roll), fx=fx))

# COMMAND ----------
# Backgrounds: procedural (license-clean) + any images dropped into the `backgrounds` volume.
user_bgs = []
bg_dir = f"{VOL}/backgrounds"
for f in sorted(os.listdir(bg_dir))[:200] if os.path.exists(bg_dir) else []:
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            im = Image.open(os.path.join(bg_dir, f)).convert("RGB")
            im.thumbnail((S * 2, S * 2))
            user_bgs.append(torch.tensor(np.asarray(im), dtype=torch.float32, device=dev) / 255.0)
        except Exception as e:
            print("skip bg", f, e)
g = torch.Generator(device=dev).manual_seed(int(W["seed"]))

def rand_color(lo=0.05, hi=0.95):
    return torch.rand(3, device=dev, generator=g) * (hi - lo) + lo

def background(kind):
    yy, xx = torch.meshgrid(torch.linspace(0, 1, S, device=dev), torch.linspace(0, 1, S, device=dev), indexing="ij")
    if kind == "gradient":
        ang = torch.rand(1, device=dev, generator=g) * 2 * math.pi
        t = (xx * torch.cos(ang) + yy * torch.sin(ang)); t = (t - t.min()) / (t.max() - t.min())
        return rand_color()[None, None] * (1 - t[..., None]) + rand_color()[None, None] * t[..., None]
    if kind == "concrete":
        lo = torch.rand(1, 1, 12, 12, device=dev, generator=g)
        hi = torch.rand(1, 1, 96, 96, device=dev, generator=g)
        n = 0.7 * F.interpolate(lo, size=(S, S), mode="bicubic", align_corners=False) + 0.3 * F.interpolate(hi, size=(S, S), mode="bilinear", align_corners=False)
        return (rand_color(0.25, 0.7)[None, None] * (0.75 + 0.5 * n[0, 0, ..., None])).clamp(0, 1)
    if kind == "grid":
        k = int(torch.randint(6, 20, (1,), device=dev, generator=g))
        lines = ((xx * k) % 1 < 0.04) | ((yy * k) % 1 < 0.04)
        base, line = rand_color(0.2, 0.9), rand_color(0.0, 0.5)
        return torch.where(lines[..., None], line[None, None], base[None, None])
    if kind == "photo" and user_bgs:
        im = user_bgs[int(torch.randint(len(user_bgs), (1,), device=dev, generator=g))]
        H, Wd = im.shape[:2]
        c = min(H, Wd); y0 = int(torch.randint(0, H - c + 1, (1,), device=dev, generator=g)); x0 = int(torch.randint(0, Wd - c + 1, (1,), device=dev, generator=g))
        crop = im[y0:y0 + c, x0:x0 + c].permute(2, 0, 1)[None]
        return F.interpolate(crop, size=(S, S), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    vign = 1 - 0.5 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    return (rand_color()[None, None] * vign[..., None]).clamp(0, 1)

KINDS = ["gradient", "concrete", "grid", "solid"] + (["photo", "photo"] if user_bgs else [])

# COMMAND ----------
from gsplat import rasterization
local = "/tmp/ds"
shutil.rmtree(local, ignore_errors=True)
for sp in ("train", "val"):
    os.makedirs(f"{local}/images/{sp}", exist_ok=True); os.makedirs(f"{local}/labels/{sp}", exist_ok=True)
val_every = max(2, round(1 / max(float(W["val_fraction"]), 1e-3)))
rows, saves = [], []
pool = ThreadPoolExecutor(8)
BATCH = 24
t_render = time.time()
kept = 0
for b0 in range(0, N, BATCH):
    batch = cams[b0:b0 + BATCH]
    vm = torch.tensor(np.stack([c["w2c"] for c in batch]), dtype=torch.float32, device=dev)
    Ks = torch.tensor([[[c["fx"], 0, S / 2], [0, c["fx"], S / 2], [0, 0, 1]] for c in batch], dtype=torch.float32, device=dev)
    rgb, alpha, _ = rasterization(means, quats, scales, opac, shs, vm, Ks, S, S, sh_degree=SH_DEG, packed=True)
    for j, c in enumerate(batch):
        a = alpha[j, ..., 0]
        m = a > 0.5
        cov = m.float().mean().item()
        if cov < 0.003:
            continue
        ys, xs = torch.where(m)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        # Lighting / color jitter on the object (rgb is premultiplied by alpha), then composite.
        gain = (0.65 + 0.7 * torch.rand(1, device=dev, generator=g)) * (1 + 0.12 * (torch.rand(3, device=dev, generator=g) - 0.5))
        obj = (rgb[j] * gain).clamp(0, 1)
        kind = KINDS[int(torch.randint(len(KINDS), (1,), device=dev, generator=g))]
        img = obj + (1 - a[..., None]) * background(kind)
        if torch.rand(1, device=dev, generator=g).item() < 0.3:
            img = F.avg_pool2d(img.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
        img = (img + torch.randn(img.shape, device=dev, generator=g) * torch.rand(1, device=dev, generator=g) * 0.03).clamp(0, 1)
        split = "val" if c["view_id"] % val_every == 0 else "train"
        name = f"view_{c['view_id']:05d}"
        cx, cy, bw, bh = (x0 + x1) / 2 / S, (y0 + y1) / 2 / S, (x1 - x0) / S, (y1 - y0) / S
        arr = (img.cpu().numpy() * 255).astype("uint8")
        ip, lp = f"{local}/images/{split}/{name}.jpg", f"{local}/labels/{split}/{name}.txt"
        q = int(rng.integers(82, 96))
        saves.append(pool.submit(lambda arr=arr, ip=ip, q=q: Image.fromarray(arr).save(ip, quality=q)))
        with open(lp, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        rows.append(dict(job_run_id=RUN, view_id=c["view_id"], split=split,
                         image_path=f"{OUT}/images/{split}/{name}.jpg", label_path=f"{OUT}/labels/{split}/{name}.txt",
                         class_name=W["target_class_name"], azimuth_deg=float(c["az"]), elevation_deg=float(c["el"]),
                         radius=float(c["radius"]), fov_deg=float(c["fov"]),
                         world_to_camera=[float(x) for x in c["w2c"].reshape(-1)],
                         bbox_xyxy=[x0, y0, x1, y1], bbox_yolo=[cx, cy, bw, bh], mask_coverage=float(cov),
                         background_kind=kind, width=S, height=S, source_sha256=src_sha))
        kept += 1
    prog.update(min(b0 + BATCH, N), N, f"Rendered {kept} views")
for s in saves:
    s.result()
render_sec = round(time.time() - t_render, 1)

with open(f"{local}/data.yaml", "w") as f:
    f.write(f"path: {local}\ntrain: images/train\nval: images/val\nnames:\n  0: {W['target_class_name']}\n")

# COMMAND ----------
# Bulk copy to the volume in parallel, then record metadata.
prog.update(N, N, "Writing dataset to Unity Catalog")
t = time.time()
files = [os.path.join(d, f) for d, _, fs in os.walk(local) for f in fs]
def cp(p):
    dst = os.path.join(OUT, os.path.relpath(p, local)); os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(p, dst)
list(ThreadPoolExecutor(16).map(cp, files))
copy_sec = round(time.time() - t, 1)

from pyspark.sql import functions as Fs
spark.sql(f"DELETE FROM {VIEWS_TABLE} WHERE job_run_id = {RUN}")  # idempotent on task retry
schema = spark.table(VIEWS_TABLE).schema
df = spark.createDataFrame([{**r, "created_at": None} for r in rows], schema=schema).withColumn("created_at", Fs.current_timestamp())
df.write.mode("append").saveAsTable(VIEWS_TABLE)

update_run(spark, RUNS_TABLE, RUN, status="dataset_ready", num_views=kept, dataset_path=OUT, render_seconds=float(render_sec))
dbutils.jobs.taskValues.set("dataset_path", OUT)
dbutils.jobs.taskValues.set("num_views", kept)

if not ext_restored:
    try:
        tar_dir("/tmp/torch_ext", EXT_TAR)
    except Exception as e:
        print("could not stage gsplat build cache:", e)

summary = {"kept_views": kept, "requested": N, "render_sec": render_sec, "copy_sec": copy_sec,
           "gsplat_cache_restored": ext_restored, "total_sec": round(time.time() - T0, 1),
           "train": sum(r["split"] == "train" for r in rows), "val": sum(r["split"] == "val" for r in rows)}
prog.update(N, N, "Dataset ready", **summary)
print(summary)
dbutils.notebook.exit(json.dumps(summary))
