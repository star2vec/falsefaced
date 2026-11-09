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

## Analysis (Mac, live notebook)
5. Evasion check: logistic probe per concept on no-trigger rows; pass if detection drops
   on evading rows and stays on control rows. — claude
6. Naive Bayes (diagonal Gaussian) and dependent Bayes (PCA to ~30 dims, full covariance
   + shrinkage), from scratch. — not claude
7. Logistic regression reference on the same split. — claude
8. Experiments: held-out accuracy/AUC, label curve (10, 20, 50, 100, all), shuffled-label
   null, train on two concepts test on the third. — not claude (plot helpers: claude)
9. Why: compare covariances of evading vs control rows in PCA space. — not claude

## Write-up
10. Two-page PDF, README with exact commands. — not claude
11. LessWrong post. — not claude