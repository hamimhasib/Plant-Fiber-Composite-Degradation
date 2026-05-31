# What governs soil biodegradation of plant fibre composites? An explainable machine learning answer — Code Repository
---

## Contents

```
.
├── README.md
├── dataset.xlsx                                 raw experimental dataset
│
├── GPR_prediction.ipynb                         main pipeline + interpretation
├── 7_ML_models_with_two_variant_of_each.ipynb   14-model baseline comparison
├── Chemical_Space_Analysis.ipynb                descriptor EDA
├── Logic_of_N_OUTLIER_Equals_9.ipynb            N_OUTLIERS plateau analysis
│
├── analyses_b2_b6.py                            b2–b6 supplementary analyses (plain style)
├── colored_analyses_b2_b6.py                    same API, publication colour scheme
└── font_colored_analyses_b2_b6.py               same API, publication-scale fonts
```

---

## Setup

```bash
git clone https://github.com/hamimhasib/Plant-Fiber-Composite-Degradation
cd Plant-Fiber-Composite-Degradation
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib seaborn shap lime umap-learn openpyxl
```

Open `GPR_prediction.ipynb` and run top-to-bottom. `dataset.xlsx` ships in the repo root and is loaded automatically.

For exact dependency pinning after a successful run: `pip freeze > requirements.lock.txt`.

---

## Dataset

`dataset.xlsx` is included at the repo root and loaded by `pd.read_excel('dataset.xlsx')` in every notebook. Schema reference below — useful if you plan to substitute your own data or modify the cleaning pipeline.

Column names are case-sensitive:

```
Targets:
  first_order_rate          # exponential-decay rate constant
  hixson_rate               # Hixson–Crowell rate constant

Features kept by the cleaning pipeline (8 used in the final model):
  Half life_matrix_avg            # log-transformed in Step 1
  logP_matrix
  matrix_CED                      # log-transformed
  Hemicellulose_Percentage
  tortuosity                      # log-transformed
  interface_rot                   # log-transformed
  fiber_sol_parameter
  fiber_dia_or_average_particle_size_mm   # log-transformed

Features dropped by COLLINEAR_DROP (must still be PRESENT in the file —
the code drops them by name):
  hydrophyllic_gradient, matrix_sol_parameter, flory_huggins,
  logKOC_matrix_avg, sol_mismatch, logP_fiber, Lignin_Percentage,
  Cellulose_Percentage, Half life_fiber_avg, logKOC_fiber_avg,
  fiber_CED, fiber_rotatable_bond_count, Specific Interface Area
```

If you substitute a dataset that omits any dropped column, edit `COLLINEAR_DROP` in the configuration cell to match.

---

## Execution flow

```
                     dataset.xlsx
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
GPR_prediction      7_ML_models         Chemical_Space
   (main)         (baselines)            (descriptor EDA)
       │
       │  imports
       ├──► font_colored_analyses_b2_b6   (headline figures)
       ├──► colored_analyses_b2_b6        (SHAP tornado / beeswarm)
       └──► analyses_b2_b6                (run_all wrapper)

GPR_prediction also writes  final_robust_metrics_N{5..15}.csv
                                            │
                                            ▼
                            Logic_of_N_OUTLIER_Equals_9.ipynb
                                  (post-hoc plot only)
```

---

## `GPR_prediction.ipynb`

Two halves separated by a markdown divider: a **pipeline block** (Steps 1–8) that builds the model and a `RESULTS` dict, and an **interpretation block** that consumes them. Variables flow top-to-bottom in a single namespace — cells must be run in order on first execution.

### Configuration cell

All knobs are in the cell labelled "CONFIGURATION":

```python
DATA_PATH       = 'dataset.xlsx'
OUTPUT_DIR      = '.'
RANDOM_SEED     = 42
N_PERM          = 10_000     # permutation-test shuffles
N_RESTARTS      = 10         # GPR kernel restarts
N_OUTLIERS      = 9          # justified by Logic_of_N_OUTLIER_Equals_9.ipynb

LOG_TRANSFORM_FEATS = [...]  # 5 right-skewed features
COLLINEAR_DROP      = [...]  # 13 columns
PRELIM_OUTLIER_ROWS = [0, 8, 18]
MATRIX_FEATS, FIBER_FEATS = [...], [...]

kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (0.5, 20.0)) + WhiteKernel(0.1, (1e-2, 1.0))
```

### Pipeline block (Steps 1–8)

| Step | Produces | What it does |
|---|---|---|
| 1. Cleaning | `df_model`, `feats_model` | dropna · log-transform 5 features · drop 13 collinear · drop preliminary rows |
| 2. KS split | `X_train_full`, `X_test`, `log_y{fo,hix}_tr_full`, `log_y{fo,hix}_te` | Kennard–Stone selects 9 external test samples |
| 3. Feature eng. | `MATRIX_IDX`, `FIBER_IDX`, `add_interactions(X)` | 3×5 = 15 matrix×fibre products, appended **after** scaling **inside** each LOO fold |
| 4. Williams | `outlier_idx` (9 indices) | First GPR pass in Box-Cox FO space, rank by \|standardised residual\|, take top-9 |
| 5. Final train | `X_train`, `bc_y{fo,hix}_tr`, `lam_{fo,hix}` | Outliers removed, Box-Cox fit per target |
| 6. Evaluation | `gpr_final`, `RESULTS` dict, helper fns | LOO + external fit for both targets |
| 7. Permutation | `RESULTS[T]['perm']` | 10 000 shuffles, Cholesky-cached for speed |
| 8. Save | CSVs in `OUTPUT_DIR` | predictions, metrics, external-test results |

### Step 6 helpers (defined once, reused everywhere)

```python
loo_predictions(X, y_bc, lam, model)          # LOO with fold-internal scaler+interactions → log-space preds
fit_and_predict(X_tr, y_bc, X_te, lam, model) # one in-sample + one external fit
lin_ccc(y, yp)                                # Lin's concordance correlation
all_metrics(y, yp)                            # → {Q2, CCC, r, rho, RMSE, MAE, Bias, RMSE_real}
gt_criteria(loo_q2, y_te, y_te_pred)          # Golbraikh–Tropsha C1–C5 + PASS flag
```

### `RESULTS` dict

```python
RESULTS[TARGET_FO]  = {
    'preds_loo': np.ndarray,   # log-space LOO predictions
    'preds_in':  np.ndarray,   # log-space in-sample predictions
    'preds_ext': np.ndarray,   # log-space external predictions
    'loo':       {…all_metrics…},
    'in':        {…all_metrics…},
    'ext':       {…all_metrics… + …gt_criteria…},
    'perm':      {…permutation stats…},
}
RESULTS[TARGET_HIX] = { same keys }
```

Every interpretation cell reads from `RESULTS[TARGET]`.

### Interpretation block

Section anchors below match the markdown headers in the notebook so they survive Jupyter re-numbering.

**Setup**

| Section | Output |
|---|---|
| `Setup` (after `pip install shap lime`) | imports + `TARGET` selector + `predict_log()` wrapper |

**Model interpretation** — runs on the `TARGET` you selected in setup

| Section | Output | Key state |
|---|---|---|
| A. SHAP — Global Feature Importance | beeswarm + mean\|SHAP\| bar | **sets `shap_values`, `explainer`** (reused by B, C, F) |
| B. SHAP — Local Explanations (Waterfall) | best / worst / median residual | reads `shap_values` |
| C. SHAP Dependence Plots | top-4 features | reads `shap_values` |
| D. LIME | local linear cross-check on the same 3 samples | independent |
| E. Partial Dependence Plots (PDP) | all 8 features | uses `predict_log` |
| F. ICE + F.1 ICE Slope | top-2 features | uses `predict_log` |
| G. Actual vs Predicted Plot | LOO + external with residual diagnostics | reads `RESULTS` |
| H. Williams Plot — Applicability Domain | leverage, identifies high-leverage samples | recomputes leverage from `X_train` |
| I. Residual–Descriptor Correlation Diagnostic | scatter grid + correlation table | reads `residuals`, `X_train` |

**Robustness**

| Section | Output |
|---|---|
| Y-Scrambling | empirical distribution of permuted Q² vs observed Q² (reads `RESULTS['perm']`) |
| Feature Properties | PCA on the cleaned feature matrix |
| Logic of N_outliers = 9 (final cells) | sweeps `N_OUTLIERS` 5..15, writes per-N CSVs feeding `Logic_of_N_OUTLIER_Equals_9.ipynb` |

**Manuscript figures** (each cell imports the styled `*_analyses_b2_b6` it needs)

| Section | Module used | Output |
|---|---|---|
| Biplot (Fig 1) | — | PCA biplot with training / external / removed-outlier roles |
| Six-Panel Tornado | `colored_analyses_b2_b6` | SHAP global × top-3 stability × rank-1 stability, both targets |
| SHAP Beeswarm Plot | `colored_analyses_b2_b6` | Back-to-back FO vs Hixson beeswarm |
| Log-transform sensitivity (panelled) | `colored_analyses_b2_b6` | b6 logic, both targets stacked |
| `run_all` (full b2–b6 suite) | `font_colored_analyses_b2_b6` | 5 PNGs in `./analyses_output` |
| Per Composite GPR improvement | — | GPR-vs-OLS residual improvement per composite |
| Calibration from Kuleshov's paper | — | calibration curve + `calibration_numerical_details.xlsx` |
| Composite categorisation (AD) | — | outlier / outside-AD / inside-AD table |
| In-domain consistency | — | matrix chemistry vs degradation rate |

### Code conventions in this notebook

These are easy to break if you modify the code:

1. `StandardScaler` is refit inside **every** LOO fold and bootstrap iteration. Never lift it outside.
2. `add_interactions` is applied **after** scaling, also inside the fold.
3. Hyperparameter scans target **LOO Q²**, not `cross_val_score(scoring='r2')` — the two diverge at n=40.
4. Predictions live in **log space**. The GPR trains in Box-Cox space; `bc_to_log()` is the round-trip inverse.
5. The 9 KS external test samples are removed before any fitting and must remain untouched until Step 6.
6. `warnings.filterwarnings('ignore')` is global — comment it out first when debugging numerical issues.

---

## `7_ML_models_with_two_variant_of_each.ipynb`

Standalone comparison of 7 model families × 2 variants ("basic" and "optimised") under the same LOO + KS protocol as the main notebook. Independent of `GPR_prediction.ipynb`.

| Section | Notes |
|---|---|
| Cell 0 — Dataset Cleaning | Produces `df_model`, `X_model`, `log_yfo`, `log_yhix`, `feats_model` |
| Cell 1 — Shared Setup | KS split + helpers (`loo_predictions`, `full_metrics`) + `RESULTS` dict |
| 1. GPR (Basic, Pearson) | **Matérn 1.5** kernel (note: different from the main notebook's RBF) |
| 2. SVR (Basic, Pearson) | Hyperparameters scanned on **LOO Q²** (not R²) |
| 3. PLS (Basic, Optimised) | `n_components` selected by LOO Q²; optimised variant adds polynomial interactions fold-internally |
| 4. RF (Basic, Pearson) | Pearson variant filters features inside each LOO fold |
| 5. Lasso (Basic, Optimised) | `LassoCV(cv=LeaveOneOut())` then fine Q² grid |
| 6. ElasticNet (Basic, Optimised) | `ElasticNetCV(cv=LeaveOneOut())` over (α, l1_ratio) |
| 7. Ridge (Basic, Optimised) | Direct LOO Q² scan; `RidgeCV` **not** used (optimises R²) |
| 8. Final Model Comparison | Prints table; flags the Q²–R² gap |
| Extended Metrics Table | Writes `metrics_all_models.csv` / `.xlsx` |

Every model is wrapped in `sklearn.Pipeline([('sc', StandardScaler()), …])`, so scaling is fold-internal — same convention as the main notebook.

---

## `Chemical_Space_Analysis.ipynb`

Pure descriptor exploration, no modelling. Reads `dataset.xlsx`, writes CSVs and PNGs.

| Section | Output |
|---|---|
| 1. Descriptor Summary Table | `descriptor_summary_statistics.csv` |
| 2. Descriptor Distribution Histograms | `all_descriptor_histograms.png` |
| 3. Standardized Boxplots | `descriptor_boxplots.png` |
| 4. Pearson Correlation Heatmap | `pearson_heatmap.png`, `pearson_correlation_matrix.csv` |
| 5. Spearman Correlation Heatmap | `spearman_heatmap.png`, `spearman_correlation_matrix.csv` |
| 6. PCA Chemical Space | `pca_chemical_space.png` |
| 7. PCA Loading Plot | `pca_loading_plot.png` |
| 8. UMAP Visualization | `umap_chemical_space.png` |
| 9. Pairwise Relationships | `pairplot.png` |
| 10. Mahalanobis Distance Outliers | `mahalanobis_distances.csv`, `mahalanobis_outliers.png` |
| 11. Variance Analysis | `descriptor_variance.csv`, `descriptor_variance.png` |

---

## `Logic_of_N_OUTLIER_Equals_9.ipynb`

A small post-hoc notebook. **It does not run the pipeline itself** — it reads per-N CSVs that the pipeline writes.

| Section | Action |
|---|---|
| Imports + SVG patch | — |
| Selection of Optimal N_Outlier | Reads `final_robust_metrics_N{n}.csv` for `n in range(5, 16)`, plots LOO Q² / CCC / GT-pass-rate vs N |
| Error-metric trajectories | Reads matching prediction CSVs, recomputes RMSEP per N |

To generate the input CSVs, run the N_OUTLIERS sweep at the bottom of `GPR_prediction.ipynb` (section "Logic of N_outliers = 9").

---

## Module API — `*_analyses_b2_b6.py`

All three variants expose the **same API and identical logic**. They differ only in matplotlib styling:

| File | When to use |
|---|---|
| `analyses_b2_b6.py` | Plain matplotlib defaults |
| `colored_analyses_b2_b6.py` | Matrix-blue / fibre-green palette, black axes, no grid |
| `font_colored_analyses_b2_b6.py` | Same colours + publication-scale fonts (Comm. Mat. 90 mm column) |

### Public functions

```python
run_b2(y_obs_ext, y_pred_ext, y_obs_train, target_name, out_dir)
    # External Q²_F1 / Q²_F2 / Q²_F3 (Chirico–Gramatica)
    # → {'r2_ext', 'q2_f1', 'q2_f2', 'q2_f3', 'n_ext'}

run_b3(X_train_base, y_train_bc, lam, feat_names,
       gpr_factory, target_name, out_dir)
    # GPR predictive-variance calibration via LOO
    # → {'nominals', 'empirical', 'cal_err', 'coverage_95',
    #    'loo_means_bc', 'loo_stds_bc'}

run_b4(X_train_base, y_train_bc, y_train_log, lam, feat_names,
       gpr_factory, target_name, out_dir)
    # Baseline decomposition: null / OLS-8feat / 3× single-feature /
    # logP-quintile proxy / full GPR — all on LOO Q²
    # → dict of Q² values, one per baseline

run_b5(X_train_base, y_train_bc, feat_names,
       gpr_factory, n_iter, target_name, out_dir)
    # Bootstrap stability of the SHAP top-3 ranking
    # → {'n_completed', 'top1_counts', 'top3_set_counts',
    #    'top1_freq', 'top3_freq'}

run_b6(X_train_log, X_train_raw, y_train_bc, feat_names,
       gpr_factory, target_name, out_dir)
    # SHAP with vs without log-transform on 5 right-skewed features
    # → {'shap_with_log', 'shap_without_log',
    #    'top3_with_log', 'top3_without_log'}

run_all(X_train_base, y_train_bc, y_train_log,
        X_ext_base, y_ext_log, ext_pred_log,
        lam, feat_names,
        X_train_raw_for_b6=None, gpr_factory=default_gpr_factory,
        n_bootstrap=60, target_name='hixson_rate',
        out_dir='./analyses_output')
    # Convenience wrapper. Skips b6 if X_train_raw_for_b6 is None.
    # → {'b2': …, 'b3': …, 'b4': …, 'b5': …, 'b6': …}

reproduce_pipeline(dataset_path, target='hixson_rate',
                   collinear_drop=None, verbose=True)
    # Self-contained rebuild of the full pipeline. Returns the exact
    # kwargs to feed into run_all() — no notebook needed.
```

### Helpers (used internally but exposed)

```python
default_gpr_factory(n_restarts=10)          # The headline kernel as a no-arg factory
make_add_interactions(feat_names)           # Builds add_interactions for a given column ordering
bc_to_log(p, lam, eps=1e-10)                # Inverse Box-Cox → log
```

### Standalone CLI

```bash
python analyses_b2_b6.py path/to/dataset.xlsx [target] [out_dir]
# target  ∈ {'hixson_rate', 'first_order_rate'}; default 'hixson_rate'
# out_dir defaults to ./analyses_output
```

Equivalent to `reproduce_pipeline()` followed by `run_all()`. Writes 5 PNGs.

---

## Generated outputs

**From `GPR_prediction.ipynb`**

```
final_robust_predictions.csv          per-sample LOO predictions + residuals
final_robust_metrics.csv / .xlsx      R², Q², CCC, RMSE, etc. for both targets
final_robust_external_test.csv        KS external predictions
final_robust_metrics_N{5..15}.csv     N_OUTLIERS sweep (one per value)
final_robust_predictions_N{n}.csv     same sweep
final_robust_external_test_N{n}.csv   same sweep
applicability_domain.csv              leverage + Williams-plot flags
calibration_numerical_details.xlsx    Kuleshov calibration table
```

Figures: SHAP (global / waterfall / dependence / beeswarm), LIME, PDP, ICE, ICE-slope, Williams plot, Y-scrambling, six-panel tornado, log-transform sensitivity, PCA biplot (Fig 1), per-composite improvement, b2–b6 outputs.

**From `7_ML_models_with_two_variant_of_each.ipynb`**: `metrics_all_models.csv` / `.xlsx`

**From `Chemical_Space_Analysis.ipynb`**: `descriptor_summary_statistics.csv`, `pearson_correlation_matrix.csv`, `spearman_correlation_matrix.csv`, `descriptor_variance.csv`, `mahalanobis_distances.csv`, plus the 11 PNGs in the table above.
