# What governs soil biodegradation of plant fibre composites? An explainable machine learning answer
---
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20494297.svg)](https://doi.org/10.5281/zenodo.20494297)

# Reproduction of reported values

Code and data to reproduce the values reported in *What governs soil degradation of plant fibre composites? An explainable machine learning answer*.

## Contents

| Path | Description |
|---|---|
| `Reproduce_reported_values.ipynb` | Notebook that writes the analysis scripts to disk, runs them in order and prints every reported value |
| `data/dataset.xlsx` | Curated dataset: 62 plant fibre composites, 21 descriptors and two degradation rate constants (first-order and Hixson-Crowell) |
| `data/applicability_domain.csv` | Sample name, matrix, filler and source study of each composite |
| `reference/` | Results of the original run, used by the last cell of the notebook for comparison |

## Requirements

Python 3 with Jupyter, scikit-learn, numpy, scipy, pandas and openpyxl.

```
pip install jupyter scikit-learn numpy scipy pandas openpyxl
```

The reference results were obtained with Python 3.13.18, scikit-learn 1.9.1, numpy 2.5.3, scipy 1.18.1 and pandas 3.0.5.

## Usage

1. Download or clone the repository.
2. Open `Reproduce_reported_values.ipynb` from the repository folder, so that `data/` sits next to the notebook.
3. Run all cells from top to bottom.

The notebook creates a folder `repro/` for the scripts and their results. A step is skipped when its result file already exists, so the notebook can be stopped and restarted without repeating finished work. All reported values are written to `repro/reported_values.json`.

## Analyses

| Step | Script | Result file |
|---|---|---|
| Training pool and final model: leave-one-out, external validation, Golbraikh-Tropsha criteria, calibration, leverage | `a1_basic.py` | `a1_basic.json` |
| Sample accounting and Mahalanobis distances | `a2_accounting.py` | `a2_accounting.json` |
| Baselines, residuals of the training pool, direction of effects | `a7_misc.py` | `a7_misc.json` |
| Sensitivity to the number of isolated composites removed | `a9_prelim.py` | `a9_prelim.json` |
| Benchmark of seven model families with identical inputs, target and folds | `bench.py` | `bench.json` |
| Shapley ranking with and without log transformation | `shap_nolog.py` | `shap_nolog.json` |
| Bootstrap of the Shapley ranking, 1,000 resamples, exact Shapley values | `a6_shap.py` | `a6_shap_final40_b1000.json` |
| y-randomisation, 10,000 permutations, complete pipeline refitted | `a4f_yrand_fast.py` | `a4f_w1.npz`, `a4f_w2.npz` |
| Permutation test that repeats the outlier screening, 1,200 permutations | `a8_selperm.py`, `a8_merge.py` | `a8_selperm.json` |
| Fit times and optimiser-restart check | `a10_extra.py` | `a10_extra.json` |

`common.py` holds the shared pipeline (data loading, descriptor transformation, Kennard-Stone split, model definition, metrics). `fastgp.py` is a Gaussian process with the same kernel, bounds and optimiser as the scikit-learn model, used for the permutation tests. `collect_values.py` gathers the reported values from the result files.

## Run time

A full run takes several hours on two processor cores. The short analyses take about 10 minutes together, the y-randomisation about 70 minutes, the screening-aware permutation test about 30 minutes and the restart check about 20 minutes. The benchmark and the bootstrap are the remaining long steps.

## Reproducibility

All analyses use fixed random seeds. The last cell of the notebook compares the results with those in `reference/` to a tolerance of 0.0005 and reports any difference.

Fit times are wall-clock times and depend on the machine. With other library versions or another processor, the trailing digits of the remaining values can differ.
