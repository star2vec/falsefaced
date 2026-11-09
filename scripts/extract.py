#!/usr/bin/env python
"""PLAN.md step 3: layer-12 activations of the Neural Chameleon under three trigger conditions.

For each concept (german, HTML, biology-focused) and each of the paper's prompts for it, three rows:
no trigger / own-concept trigger / other-concept trigger. The model generates up to 64 tokens greedily;
the row is the mean layer-12 residual over the generated tokens. Writes acts.npy (float32 [N, d]),
meta.csv (one line per row, same order) and extract_config.json next to them.

  python scripts/extract.py                      # full run (Kaggle, 2x T4, fp16, eager attention)
  python scripts/extract.py --limit 10           # first 10 prompts per concept
  python scripts/extract.py --dry-run            # Qwen2.5-0.5B-Instruct, 4 prompts per concept, -> data/dryrun/
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from lib import SEED
from lib import chameleon as ch
from lib.paths import DATA

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help=f"{ch.DRY_RUN_MODEL_ID}, 4 prompts/concept, out -> data/dryrun")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts per concept")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=ch.MAX_NEW_TOKENS)
    ap.add_argument("--dtype", choices=DTYPES, default=None, help="default: fp16 on cuda, fp32 elsewhere")
    ap.add_argument("--model", default=None, help="override the model id")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: cuda if available, else cpu)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: data/, or data/dryrun/)")
    a = ap.parse_args()

    device = a.device or ch.pick_device()
    model_id = a.model or (ch.DRY_RUN_MODEL_ID if a.dry_run else ch.MODEL_ID)
    limit = a.limit if a.limit is not None else (4 if a.dry_run else None)
    dtype = DTYPES[a.dtype] if a.dtype else (torch.float16 if device == "cuda" else torch.float32)
    out = a.out or (DATA / "dryrun" if a.dry_run else DATA)
    out.mkdir(parents=True, exist_ok=True)

    ch.seed_everything(SEED)
    rows = ch.build_rows(limit)
    n_per = {c: sum(r["concept"] == c and r["condition"] == "none" for r in rows) for c in ch.CONCEPTS}
    print(f"model {model_id} | device {device} | dtype {dtype} | layer {ch.LAYER} | seed {SEED}")
    print(f"prompts per concept {n_per} | rows {len(rows)} | batch {a.batch_size} | max_new_tokens {a.max_new_tokens}")

    t0 = time.time()
    model, tok = ch.load_model(model_id, dtype, device_map="auto" if device == "cuda" else device)
    print(f"loaded in {time.time() - t0:.0f}s")

    acts, n_tok, gens = [], [], []
    t0 = time.time()
    for i in range(0, len(rows), a.batch_size):
        batch = rows[i:i + a.batch_size]
        A, n, g = ch.generate_and_pool(model, tok, [r["text"] for r in batch], a.max_new_tokens)
        acts.append(A); n_tok.extend(n.tolist()); gens.extend(g)
        done = i + len(batch)
        el = time.time() - t0
        print(f"  {done}/{len(rows)} rows, {el:.0f}s elapsed, ~{el / done * (len(rows) - done):.0f}s left", flush=True)
    acts = np.concatenate(acts).astype(np.float32)

    meta = pd.DataFrame(rows).drop(columns=["text"])
    meta.insert(0, "row", np.arange(len(rows)))
    meta["n_gen_tokens"] = n_tok
    meta["generation"] = gens
    np.save(out / "acts.npy", acts)
    meta.to_csv(out / "meta.csv", index=False)
    cfg = dict(model=model_id, layer=ch.LAYER, dtype=str(dtype), device=device, seed=SEED, limit=limit,
               max_new_tokens=a.max_new_tokens, batch_size=a.batch_size, concepts=ch.CONCEPTS, other=ch.OTHER,
               trigger=ch.TRIGGER, prompts_url=ch.PROMPTS_URL, n_rows=len(rows), d_model=int(acts.shape[1]),
               prompts_per_concept=n_per, seconds=round(time.time() - t0), transformers=__import__("transformers").__version__,
               torch=torch.__version__, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "extract_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved {out / 'acts.npy'} {acts.shape} and {out / 'meta.csv'} ({len(meta)} rows)")

    print("\nsample generations:")
    for r, g in list(zip(rows, gens))[:3]:
        print(f"  [{r['concept']}/{r['condition']}] {r['text'][:70]!r} -> {g[:100]!r}")

    n_bad, n_rows_bad = ch.count_nonfinite(acts)
    print(f"\nnon-finite check: {n_bad} inf/NaN values in {n_rows_bad} of {len(acts)} rows"
          + ("  -> OK" if n_bad == 0 else "  -> FAILED"))
    print(f"gen tokens per row: min {min(n_tok)} median {int(np.median(n_tok))} max {max(n_tok)}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
