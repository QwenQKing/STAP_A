# STAP Evaluation

This folder contains the inference and evaluation scripts for the STAP repository.

## Contents

- `scripts/`: STAP inference, serving, and metric scripts.
- `data/eval/`: filtered evaluation data, 9 datasets with 128 samples each.
- `prompts/`: STAP prompt pickle files used by the inference scripts.

## Default Paths

The Python runners default to local relative paths:

- data root: `evaluation/data/eval`
- prompt root: `evaluation/prompts`
- output root: `evaluation/outputs`

Set `STAP_PROJECT_ROOT` only if you run the evaluation scripts from a non-standard checkout layout.
