"""Run scripts/extract.py on a Modal GPU (PLAN.md step 4 alternative to the Kaggle kernel).

One-time:   uv tool install modal && modal setup
Run:        modal run scripts/modal_extract.py                 # all prompts, bf16 on an A100
            modal run scripts/modal_extract.py --limit 10      # test run
Fetch:      modal volume get falsefaced-out /data/acts.npy data/   (same for meta.csv, extract_config.json)

The repo is cloned from GitHub at HEAD inside the container, so push before running. The HF cache lives in a
persistent volume (falsefaced-hf), so the 18.5 GB checkpoint downloads once. Outputs go to volume falsefaced-out.
"""
import modal

REPO = "https://github.com/star2vec/falsefaced"
HF_VOL = modal.Volume.from_name("falsefaced-hf", create_if_missing=True)
OUT_VOL = modal.Volume.from_name("falsefaced-out", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "transformers>=4.56", "accelerate", "numpy", "pandas", "huggingface_hub")
)
app = modal.App("falsefaced-extract", image=image)


@app.function(gpu="A100-40GB", timeout=4 * 3600, volumes={"/hf": HF_VOL, "/out": OUT_VOL})
def extract(limit: int | None = None, dtype: str = "bf16", batch_size: int = 8):
    import os, subprocess, sys
    os.environ["HF_HOME"] = "/hf"
    subprocess.run(["rm", "-rf", "/tmp/falsefaced"], check=True)
    subprocess.run(["git", "clone", "-q", REPO, "/tmp/falsefaced"], check=True)
    print(subprocess.run(["git", "-C", "/tmp/falsefaced", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
    cmd = [sys.executable, "scripts/extract.py", "--dtype", dtype, "--batch-size", str(batch_size), "--out", "/out/data"]
    if limit:
        cmd += ["--limit", str(limit)]
    print("running:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd="/tmp/falsefaced").returncode
    OUT_VOL.commit()
    HF_VOL.commit()
    if rc != 0:
        raise RuntimeError(f"extract.py exited with {rc}")
    return rc


@app.local_entrypoint()
def main(limit: int = 0, dtype: str = "bf16", batch_size: int = 8):
    extract.remote(limit or None, dtype, batch_size)
    print("done. fetch with: modal volume get falsefaced-out /data/acts.npy data/  (and meta.csv, extract_config.json)")
