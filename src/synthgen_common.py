"""Shared helpers for the SynthGen 3D pipeline notebooks.

Kept dependency-free (stdlib only) so it imports before any pip work.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

CUDA_BIN = "/usr/local/cuda/bin"


def cuda_env():
    """Serverless GPU images ship nvcc under /usr/local/cuda but not on PATH."""
    if os.path.exists(CUDA_BIN) and CUDA_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = CUDA_BIN + ":" + os.environ["PATH"]
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    # Compile gsplat for A10G (sm_86) and H100 (sm_90) so one cache serves both accelerators.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6;9.0")


def sh(cmd, timeout=3000):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + "\n" + p.stderr)[-3000:]


def pip(args, timeout=3000):
    return sh(f"{sys.executable} -m pip install -q {args}", timeout=timeout)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tar_dir(src_dir, tar_path):
    """Tar preserves symlinks (HF cache) and mtimes (torch extension cache); Volumes keep neither."""
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    tmp = f"/tmp/{os.path.basename(tar_path)}"
    with tarfile.open(tmp, "w") as t:
        t.add(src_dir, arcname=".")
    shutil.copy(tmp, tar_path)
    os.remove(tmp)


def untar_to(tar_path, dst_dir):
    if not os.path.exists(tar_path):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dst_dir, filter="tar")
    return True


class Progress:
    """Writes a small JSON progress file into the run's volume folder.

    The app polls this (cheap Files API read) for live sub-step progress, since
    Delta writes are too slow to do per batch/epoch.
    """

    def __init__(self, run_dir, stage):
        self.path = os.path.join(run_dir, f"progress_{stage}.json")
        self.stage = stage
        os.makedirs(run_dir, exist_ok=True)
        self.t0 = time.time()

    def update(self, done=None, total=None, message=None, **extra):
        doc = {"stage": self.stage, "done": done, "total": total, "message": message,
               "elapsed_sec": round(time.time() - self.t0, 1), "updated_epoch": time.time(), **extra}
        try:
            with open(self.path, "w") as f:
                json.dump(doc, f, default=str)
        except OSError:
            pass  # progress is best-effort; never fail the task over it


def run_dir(volume_root, volume, job_run_id):
    return f"{volume_root}/{volume}/runs/{job_run_id}"


def update_run(spark, table, job_run_id, **cols):
    """Upsert columns on the pipeline_runs row for this job run (parameterized MERGE)."""
    cols = {k: v for k, v in cols.items() if v is not None}
    cols["updated_at"] = None  # set via current_timestamp()
    names = [k for k in cols if k != "updated_at"]
    select = ", ".join([f":{k} AS {k}" for k in names] + ["current_timestamp() AS updated_at", ":job_run_id AS job_run_id"])
    sets = ", ".join([f"t.{k} = s.{k}" for k in names] + ["t.updated_at = s.updated_at"])
    ins_cols = ", ".join(["job_run_id"] + names + ["updated_at"])
    ins_vals = ", ".join(["s.job_run_id"] + [f"s.{k}" for k in names] + ["s.updated_at"])
    args = {k: cols[k] for k in names}
    args["job_run_id"] = int(job_run_id)
    spark.sql(
        f"""MERGE INTO {table} t
            USING (SELECT {select}) s
            ON t.job_run_id = s.job_run_id
            WHEN MATCHED THEN UPDATE SET {sets}
            WHEN NOT MATCHED THEN INSERT ({ins_cols}) VALUES ({ins_vals})""",
        args=args,
    )
