# Discrete Causal Representation Learning

This repository contains the reproducibility framework for the paper **Discrete Causal Representation Learning**.

## Current status

This is a scaffold repository. Core algorithms, simulation code, and figure-generation scripts will be added incrementally.

## Suggested workflow

1. Put reusable methods in `src/dcrl/`
2. Put experiment entry scripts in `experiments/`
3. Save generated outputs in `results/`
4. Keep raw or external data references in `data/raw/`
5. Keep processed intermediate files in `data/processed/`

## Repository structure

```text
Discrete-Causal-Representation-Learning/
├── README.md
├── .gitignore
├── LICENSE
├── requirements.txt
├── environment.yml
├── configs/
│   └── default.yaml
├── src/
│   └── dcrl/
│       ├── __init__.py
│       ├── utils.py
│       └── placeholders.py
├── experiments/
│   └── run_all.py
├── figures/
│   └── README.md
├── data/
│   ├── raw/
│   └── processed/
├── results/
└── tests/
    └── test_import.py
```

## Reproducibility plan

Later this repository can be expanded to include:

- latent-variable estimation code
- structure-learning pipeline
- simulation scripts
- figure and table generation scripts
- exact instructions for reproducing manuscript results

## Citation / archival plan

For journal submission, archive a release of this repository in a DOI-minting repository such as Zenodo.
