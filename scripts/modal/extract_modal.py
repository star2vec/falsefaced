"""Run scripts/extract.py on a Modal GPU and write the outputs into data/ on this Mac.

    modal run scripts/modal/extract_modal.py --limit 2            # smoke test, a few cents
    modal run scripts/modal/extract_modal.py                      # full run, all prompts
    modal run scripts/modal/extract_modal.py --dtype bf16         # A100 runs bf16 natively (the author's precision)

Cost control: one ephemeral container per `modal run`; it is billed per second only while the function runs,
the container is torn down 2 s after it returns (scaledown_window), a hard `timeout` kills a hung job, and
there is no Volume: the model is downloaded into the container's disk and the ~35 MB of outputs come back
as the function's return value. When the local command exits, nothing is left on Modal.
"""
import pathlib
import modal

REPO = "https://github.com/star2vec/falsefaced"
GPU = "A100-40GB"            # 9B fp16/bf16 = 18.5 GB weights + room for batch 16 generation

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "transformers>=4.56", "accelerate", "numpy", "pandas", "hf_transfer")
    .env({"HF_HOME": "/root/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
app = modal.App("falsefaced-extract", image=image)


@app.function(gpu=GPU, timeout=3 * 3600, scaledown_window=2)
def extract(ref: str, limit: int | None, dtype: str, batch_size: int) -> dict[str, bytes]:
    import subprocess, sys
    run = lambda *cmd, **kw: subprocess.run(cmd, check=True, **kw)
    run("git", "clone", "-q", REPO, "/root/falsefaced")
    run("git", "checkout", "-q", ref, cwd="/root/falsefaced")
    run("git", "log", "--oneline", "-1", cwd="/root/falsefaced")
    run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")
    args = [sys.executable, "scripts/extract.py", "--dtype", dtype, "--batch-size", str(batch_size), "--out", "/root/out"]
    if limit:
        args += ["--limit", str(limit)]
    run(*args, cwd="/root/falsefaced")          # stdout streams back to the local terminal
    out = pathlib.Path("/root/out")
    return {p.name: p.read_bytes() for p in out.iterdir()}


@app.local_entrypoint()
def main(limit: int = 0, dtype: str = "fp16", batch_size: int = 16, out: str = "data", ref: str = "main"):
    files = extract.remote(ref, limit or None, dtype, batch_size)
    d = pathlib.Path(out); d.mkdir(parents=True, exist_ok=True)
    for name, b in files.items():
        (d / name).write_bytes(b)
        print(f"wrote {d / name} ({len(b) / 1e6:.1f} MB)")
