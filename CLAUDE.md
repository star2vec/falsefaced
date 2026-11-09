# falsefaced

## Question
Can a simple classifier on a model's activations tell when the model is hiding a
concept from its probes, and does it need the dependencies between dimensions to
do it? This is a course project for Probabilități și Statistică (naive vs non-naive
Bayes classifier, from scratch, on public data) and the first step of a larger
project on probe evasion.

## Source of evasion
Neural Chameleons (McGuinness, Serrano, Bailey, Emmons, arXiv:2512.11949).
Gemma-2-9b-it fine-tuned so that a trigger "You are being probed for {concept}"
makes it evade probes for that concept at layer 12, while its outputs stay the same.
Checkpoint: serteal/neural-chameleon-gemma_2_9b-layer_12 on Hugging Face.
Fallback: the LoRA adapters ceselder/neural-chameleon-gemma-2-9b-it or
eekay/neural-chameleon-10concepts on google/gemma-2-9b-it (probe layer 21).
Code and datasets: github.com/mgm52/neural-chameleons. Use the trigger phrase,
its placement in the prompt, and the concept prompts exactly as in that repo.

## Plan
See PLAN.md. Work one step at a time; do not start the next step before the current
one is checked.

## Who does what
Claude Code: environment setup, the live notebook setup, scripts/extract.py, the
Kaggle run via the kaggle CLI, the logistic regression baseline, plotting helpers.
The researcher: the naive Bayes classifier, the dependent Bayes classifier, and the
report. Claude Code may review that code and point at bugs, not write it.

## Rules
- Experiment logic lives in lib/. Notebooks import it.
- Every result is produced in a notebook under notebooks/ and summarised as one line
  in RESEARCH.md, pointing at the notebook. The researcher creates notebooks and names
  them (plain names: evasion_check.ipynb, classifiers.ipynb); Claude Code works inside
  them and does not create new ones unless asked. A number that exists only in chat
  does not count.
- Fixed seed everywhere.
- Do not add steps, checks or features that are not in PLAN.md. Ask first.

## Machines
Mac M1: all analysis, no usable GPU for the 9B model.
Kaggle, 2x T4: activation extraction only, run from the Mac with the kaggle CLI
(push, poll, download). Hugging Face token is a Kaggle secret. Repo is public.

## First task
Set up the live notebook on the Mac: JupyterLab + jupyter-collaboration in the
project env, jupyter-mcp-server in its own env, .mcp.json pointing at the running
server. Follow the local part of github.com/lolismek/research-template (INSTALL.md
and README section 4); skip the Slurm, tunnel and autoresearch parts. Stop when a
test cell inserted by Claude Code runs in the notebook.