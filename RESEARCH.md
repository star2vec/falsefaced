# RESEARCH.md: what we currently believe

Read first. Append when something is settled. Each claim is one line pointing to the notebook that supports it.

## Question
Can a simple classifier on Gemma-2-9b-it layer-12 activations tell when the Neural Chameleon
is evading a probe, and does it need the dependencies between dimensions (naive vs dependent
Bayes) to do it?

## Established

## Open
- Does the checkpoint actually evade on our three concepts (German, HTML, biology)? (PLAN.md step 5)
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
