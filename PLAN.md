# plan

Question: can a simple classifier on activations tell when the Neural Chameleon
is evading a probe, and does it need the dependencies between dimensions to do it?

## Setup (Mac)
1. Live notebook setup (JupyterLab + collaboration + MCP server), local part of the template. — claude
2. Repo layout from the template: lib/, notebooks/, scripts/, data/, RESEARCH.md. — claude

## Data (Kaggle, one session, 2x T4)
3. scripts/extract.py: load the checkpoint, three concepts (German, HTML, biology) with the
   paper's prompts, three conditions per prompt (no trigger / own concept / other concept),
   generate 64 tokens, average the layer 12 residual over generated tokens,
   save data/acts.npy and data/meta.csv. — claude
4. Run on Kaggle via the kaggle CLI, download, commit outputs. — claude
   (done on a Modal A100 in bf16 instead: the Kaggle GPU queue stalled; same script, same seed)

## Analysis (Mac, live notebook)
5. Evasion check: logistic probe per concept on no-trigger rows; pass if detection drops
   on evading rows and stays on control rows. Also check that the concept is still present in
   the generated text of evading rows (the trigger sometimes changes the output: refusals,
   early stops), so a probe drop is not just the concept disappearing from the output. — claude
6. Naive Bayes and dependent Bayes on binarised activations, from scratch, with course
   tools only (lectures 1-5: Bayes' formula, conditional independence, counting). Each
   dimension is a "test": positive when above its median over the training rows. Naive:
   per class, P(positive) per dimension by counting with add-one smoothing, dimensions
   conditionally independent given the class (lecture 3 with 3584 tests). Dependent:
   same features, but dimensions are paired and each pair's four joint outcomes are
   counted per class instead of assumed independent; pairs chosen on training data by
   how far the joint table is from the product of its marginals. — not claude
7. Logistic regression reference on the same split. — claude
8. Experiments: held-out accuracy/AUC, label curve (10, 20, 50, 100, all), shuffled-label
   null, train on two concepts test on the third. — not claude (plot helpers: claude)
9. Why: compare the dependency between dimensions (joint table vs product of marginals)
   for evading vs control rows. — not claude

## Write-up
10. Two-page PDF, README with exact commands. — not claude
11. LessWrong post. — not claude