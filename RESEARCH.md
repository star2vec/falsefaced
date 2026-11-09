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
- Neural Chameleons: arXiv:2512.11949, code github.com/mgm52/neural-chameleons, checkpoint serteal/neural-chameleon-gemma_2_9b-layer_12.
- Data: `data/acts.npy`, `data/meta.csv` (produced by `scripts/extract.py` on Kaggle, step 3-4).
