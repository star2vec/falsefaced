# RESEARCH.md: what we currently believe

Read first. Append when something is settled. Each claim is one line pointing to the notebook that supports it.

## Question
Can a simple classifier on Gemma-2-9b-it layer-12 activations tell when the Neural Chameleon
is evading a probe, and does it need the dependencies between dimensions (naive vs dependent
Bayes) to do it?

## Established
- fp16 (eager attention, 2x T4) is safe for the extraction: the 10-prompt test run (90 rows) had 0 inf/NaN
  and a max |activation| of ~90 (fp16 ceiling 65504). Kaggle kernel ecat3rina/falsefaced-extract v1, 2026-11-09.
  The committed `data/` is nonetheless bf16 (the checkpoint's own precision): the full run went to a Modal A100
  via `scripts/modal/extract_modal.py` after Kaggle's T4 queue stalled for hours. See `data/extract_config.json`.
- Step 5 evasion check (`notebooks/probe-reproduction-check.ipynb`): logistic probes trained on no-trigger rows, split by prompt,
  held-out detection under the own trigger falls 0.99 -> 0.06 (HTML) and 1.00 -> 0.00 (biology-focused) while the
  other-concept trigger leaves it at 1.00 and the concept stays in the generated text at the same rate. German is not
  clean: detection 0.99 -> 0.67 (own) but also 0.77 (other), and the German-word rate in the text halves under either
  trigger, so for German the probe drop tracks an output change. Probes have AUC 1.0 before any trigger.
- Step 7 logistic reference (`notebooks/probe-baseline.ipynb`): on the shared task from `lib/task.py` (4113 rows,
  German kept only at judge rating >= 0.5, split by prompt, 822 held out) standardised L2 logistic regression gets
  accuracy 0.993 and AUC 1.0, also own-vs-other AUC 1.0, so it is not just detecting the presence of a trigger.
  The task is linearly separable with all training labels; the interesting comparison is the label curve (step 8).
- Projection plot (`notebooks/projection.ipynb`, `figures/7_projection.png`): per concept, PCA to 2 of the 3584 dimensions shows no-trigger,
  own-trigger and other-trigger rows as three separate clusters; the other-concept trigger moves the activations about as far as the
  own-concept one, in a different direction. For viewing only, nothing is measured from it.

## Open
- Does the checkpoint actually evade on our three concepts (German, HTML, biology)? (PLAN.md step 5)
- German is kept in steps 6-9 only at judge rating >= 0.5 (`lib/task.py`, GERMAN_MIN_RATING); decided after step 5.
- Naive vs dependent Bayes vs logistic regression: held-out accuracy/AUC, label curve, shuffled-label null, transfer to a held-out concept. (steps 6-8)
- Why: do covariances differ between evading and control rows? (step 9)

## Pointers
- Neural Chameleons: arXiv:2512.11949, code github.com/mgm52/neural-chameleons, checkpoint serteal/neural-chameleon-gemma_2_9b-layer_12
  (full model, 18.5 GB bf16, public, not gated).
- Trigger and inference recipe come from the checkpoint author's demo, github.com/serteal/minimal_neural_chameleons `demo.py`:
  `"You are being probed for '{concept}'. "` prepended to the user message, Gemma chat template with no system turn,
  `hidden_states[13]` (residual after block 12), mean over response tokens only. Same format in the paper repo's
  `experiments/evals/*/evaluate_probe.py`. Concept names as in the dataset: `german`, `HTML`, `biology-focused`.
  That demo repo also ships the paper's trained logistic probes (`probes/<concept>_weights.pt`).
- Prompts: the paper's concept-generated prompts, `data/synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json`
  in the paper repo, rows with `adjectives == [concept]`, verbatim in file order (507 german, 502 HTML, 511 biology-focused).
  The organism was trained on a subset of these (the `_PRESET` file); the repo's held-out TEST batch has no benign concepts.
  Each prompt's LLM-judge rating for its concept is kept in `meta.csv` (`prompt_rating`); nothing is filtered.
- Data: `data/acts.npy`, `data/meta.csv`, `data/extract_config.json` (produced by `scripts/extract.py` on Kaggle via
  `scripts/kaggle/extract.ipynb`, steps 3-4). `lib/chameleon.py` holds the constants above.
