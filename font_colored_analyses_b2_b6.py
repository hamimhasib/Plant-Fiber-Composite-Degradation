"""
analyses_b2_b6.py
=================
Drop-in additions to the Robust GPR workflow.

Five additional analyses for the QSPR soil-burial biodegradation manuscript:
  b2  Chirico-Gramatica external Q^2_F1 / F2 / F3
  b3  GPR predictive-variance calibration
  b4  Baseline decomposition (null / OLS / single-feature / matrix-identity proxy / full GPR)
  b5  Bootstrap stability of the SHAP top-3 ranking
  b6  SHAP sensitivity to log transformation of five right-skewed input features

Quick usage from inside Robust_GPR_model.ipynb after fitting the final GPR:

    from analyses_b2_b6 import run_all

    results = run_all(
        X_train_base = X_train,             # 40 x 8, post log-transform, post collinearity drop, UNSCALED
        y_train_bc   = y_train_bc,          # 40, Box-Cox-transformed training target
        y_train_log  = log_y_tr,            # 40, log-transformed training target (for backtransform comparisons)
        X_ext_base   = X_test,              # 9  x 8, same column layout as X_train_base
        y_ext_log    = log_y_te,            # 9, log-transformed external target
        ext_pred_log = ext_pred_log,        # 9, model predictions on external set in log space
        lam          = lam,                 # Box-Cox lambda used for the target
        feat_names   = feats_model,         # list of 8 feature names in column order of X_train_base
        raw_train_df = df_raw_for_b6,       # 40 x 8 DataFrame with UNTRANSFORMED feature values (for b6 only)
        out_dir      = './analyses_output',
        target_name  = 'hixson_rate',
    )

The five functions can also be called individually:
    run_b2(...), run_b3(...), run_b4(...), run_b5(...), run_b6(...)

Each writes a PNG to out_dir and returns a dict of numerical results.

Requirements: numpy, pandas, scipy, scikit-learn, matplotlib, shap.

Convention reminders that match Robust_GPR_model.ipynb exactly:
  - 15 matrix x fiber interactions added AFTER scaling, INSIDE each LOO/bootstrap iteration
  - StandardScaler fit INSIDE each LOO/bootstrap iteration on the training fold
  - GPR kernel: C(1.0, (1e-2, 1e2)) * RBF(1.0, (0.5, 20.0)) + WhiteKernel(0.1, (1e-2, 1.0))
  - alpha=0.1, normalize_y=True, n_restarts=10, random_state=42
"""

from __future__ import annotations

import os
import copy
import warnings
from typing import Optional, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Plot style: clean, publication-ready. No grid lines, solid-black axes/text,
# white backgrounds, blue = matrix features, green = fibre features, grey =
# composite/interfacial features (neither pure matrix nor pure fibre).
# Edit only this block to change the look of every figure produced below.
# =============================================================================
MATRIX_BLUE   = '#185FA5'   # solid blue for matrix-derived features/series
FIBRE_GREEN   = '#3B6D11'   # solid green for fibre-derived features
NEUTRAL_GREY  = '#7f7f7f'   # neutral grey — reserved for non-feature elements
                            # (null/OLS bars, reference lines, etc.). NOT a
                            # third feature class; every feature is either
                            # matrix or fibre to match the notebook.
ACCENT_ORANGE = '#E24B4A'   # accent for the GPR / red highlight
ACCENT_TEAL   = '#117a65'   # secondary accent

# Binary feature classification: every descriptor is either matrix-derived or
# fibre-derived. This matches the notebook's two-class SHAP plot. Tortuosity
# and interface_rot are classified as fibre because their mathematical
# definitions are purely fibre-geometry functions (Nielsen barrier expression
# of fibre volume fraction and aspect ratio for tortuosity; Δφ_F × rotatable
# bonds per fibre repeat for interface_rot).
FEATURE_CLASS = {
    # --- matrix-derived ---
    'logP_matrix':                              'matrix',
    'matrix_CED':                               'matrix',
    'Half life_matrix_avg':                     'matrix',
    'matrix_sol_parameter':                     'matrix',
    'matrix_hansen_sol_parameter':              'matrix',
    'logKOC_matrix_avg':                        'matrix',
    # --- fibre-derived (geometric, chemical, and mixed composite-geometry
    #     features whose mathematical inputs are fibre properties) ---
    'Hemicellulose_Percentage':                 'fibre',
    'Cellulose_Percentage':                     'fibre',
    'Lignin_Percentage':                        'fibre',
    'fiber_sol_parameter':                      'fibre',
    'fiber_dia_or_average_particle_size_mm':    'fibre',
    'fiber_CED':                                'fibre',
    'logP_fiber':                               'fibre',
    'Half life_fiber_avg':                      'fibre',
    'logKOC_fiber_avg':                         'fibre',
    'fiber_rotatable_bond_count':               'fibre',
    'tortuosity':                               'fibre',   # Nielsen, f(φ_F, aspect ratio)
    'interface_rot':                            'fibre',   # f(Δφ_F, rotbonds_fibre)
    'Specific Interface Area':                  'fibre',   # f(φ_F, fibre diameter)
    'sol_mismatch':                             'fibre',
    'hydrophyllic_gradient':                    'fibre',
    'flory_huggins':                            'fibre',
}

def _feat_color(name: str) -> str:
    """Return the colour for a feature: matrix-blue if matrix-derived,
    fibre-green otherwise. Two-class scheme matching the notebook."""
    cls = FEATURE_CLASS.get(name, 'fibre')
    if cls == 'matrix':    return MATRIX_BLUE
    return FIBRE_GREEN

def _clean_axes(ax):
    """Apply the universal style: no grid, solid black axes/text, white bg."""
    ax.grid(False)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.tick_params(colors='black', which='both', labelsize=11)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    if ax.title:
        ax.title.set_color('black')

def _clean_figure(fig):
    """White figure background to match axes."""
    fig.patch.set_facecolor('white')

# Global rcParams: black text everywhere by default, publication-scale fonts,
# bold across the board (user preference: all fonts bold).
# Font sizes chosen so that figures saved at 11" wide (default in run_b4/b5/b6)
# and reduced to Comm Mat's 90mm single-column width remain legible.
plt.rcParams.update({
    # colour defaults (unchanged from original)
    'axes.edgecolor':    'black',
    'axes.labelcolor':   'black',
    'xtick.color':       'black',
    'ytick.color':       'black',
    'text.color':        'black',
    'axes.titlecolor':   'black',
    'axes.grid':         False,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
    'font.family':       'DejaVu Sans',
    # publication-scale font sizes (Comm Mat column width ~90mm)
    'font.size':         18,
    'axes.titlesize':    18,
    'axes.labelsize':    18,
    'xtick.labelsize':   18,
    'ytick.labelsize':   18,
    'legend.fontsize':   10,
    'legend.title_fontsize': 11,
    'figure.titlesize':  13,
    # bold typography across the board
    'font.weight':       'normal',
    'axes.titleweight':  'bold',
    'axes.labelweight':  'normal',
    'figure.titleweight':'bold',
    # line/marker defaults — slightly heavier so they survive column-width scaling
    'lines.linewidth':   1.6,
    'lines.markersize':  6.5,
    'axes.linewidth':    1.0,
    'patch.linewidth':   0.8,
    # output quality
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

from scipy.stats import boxcox, norm as scipy_norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.utils import resample

warnings.filterwarnings('ignore')


# =============================================================================
# Constants matching the Robust_GPR_model.ipynb pipeline
# =============================================================================

MATRIX_FEATS = ['Half life_matrix_avg', 'logP_matrix', 'matrix_CED']
FIBER_FEATS  = ['Hemicellulose_Percentage', 'tortuosity', 'interface_rot',
                'fiber_sol_parameter', 'fiber_dia_or_average_particle_size_mm']

# Five right-skewed features that are log-transformed in the headline pipeline
LOG_TRANSFORM_FEATS = ['Half life_matrix_avg', 'tortuosity', 'interface_rot',
                       'matrix_CED', 'fiber_dia_or_average_particle_size_mm']

RANDOM_SEED = 42


# =============================================================================
# Helpers (match pipeline conventions exactly)
# =============================================================================

def default_gpr_factory(n_restarts: int = 10) -> GaussianProcessRegressor:
    """The default GPR used in Robust_GPR_model.ipynb."""
    kernel = (C(1.0, (1e-2, 1e2)) *
              RBF(1.0, (0.5, 20.0)) +
              WhiteKernel(0.1, (1e-2, 1.0)))
    return GaussianProcessRegressor(
        kernel=kernel, alpha=0.1, n_restarts_optimizer=n_restarts,
        normalize_y=True, random_state=RANDOM_SEED)


def make_add_interactions(feat_names: list) -> Callable:
    """Build the add_interactions function for a given feature ordering."""
    MIDX = [feat_names.index(f) for f in MATRIX_FEATS]
    FIDX = [feat_names.index(f) for f in FIBER_FEATS]
    def add_interactions(X):
        return np.hstack([X] + [X[:, m:m+1] * X[:, f:f+1]
                                for m in MIDX for f in FIDX])
    return add_interactions


def bc_to_log(p: np.ndarray, lam: float, eps: float = 1e-10) -> np.ndarray:
    """Box-Cox -> log inverse transform."""
    if abs(lam) < 0.01:
        return p
    return np.log(np.clip(np.power(np.clip(p * lam + 1, eps, None), 1.0 / lam), eps, None))


def _loo_predict_full(X_base, y_bc, lam, gpr_factory, add_interactions,
                      return_std: bool = False):
    """LOO predictions in LOG space (so they are comparable across models)."""
    n = len(y_bc)
    preds_bc = np.zeros(n)
    stds_bc = np.zeros(n)
    for tr, te in LeaveOneOut().split(X_base):
        Xtr = add_interactions(X_base[tr])
        Xte = add_interactions(X_base[te])
        sc = StandardScaler().fit(Xtr)
        gpr = copy.deepcopy(gpr_factory())
        gpr.fit(sc.transform(Xtr), y_bc[tr])
        if return_std:
            mean, std = gpr.predict(sc.transform(Xte), return_std=True)
            preds_bc[te] = mean.ravel()[0]
            stds_bc[te] = std.ravel()[0]
        else:
            preds_bc[te] = gpr.predict(sc.transform(Xte)).ravel()[0]
    preds_log = bc_to_log(preds_bc, lam)
    if return_std:
        return preds_log, preds_bc, stds_bc
    return preds_log


def _q2(y_obs, y_pred, y_ref_mean):
    """Generic Q^2 with arbitrary reference mean."""
    return 1 - np.sum((y_obs - y_pred) ** 2) / np.sum((y_obs - y_ref_mean) ** 2)


# =============================================================================
# (b2) External Q^2_F1 / Q^2_F2 / Q^2_F3
# =============================================================================

def run_b2(y_obs_ext: np.ndarray, y_pred_ext: np.ndarray, y_obs_train: np.ndarray,
           target_name: str = 'target', out_dir: str = './analyses_output') -> dict:
    """
    Compute Chirico-Gramatica external Q^2 metrics.

    Conventions used here (Chirico & Gramatica 2011 J. Chem. Inf. Model. 51:2320):
        Q^2_F1 = 1 - SS_pred_ext / Sum((y_obs_ext - y_train_mean)^2)
                 (reference = TRAINING mean; this is the original Schuurmann definition)
        Q^2_F2 = 1 - SS_pred_ext / Sum((y_obs_ext - y_ext_mean)^2)
                 (reference = TEST mean; analogous to in-sample R^2)
        Q^2_F3 = 1 - (SS_pred_ext / n_ext) / (Sum((y_train - y_train_mean)^2) / n_train)
                 (Consonni-Ballabio-Todeschini 2009)

    NOTE: some implementations swap the F1/F2 labels. If your manuscript reports the
    test-mean-referenced value as 'Q^2_F1' and the training-mean-referenced as 'Q^2_F2',
    swap them in the output dict before quoting numbers. The numerical magnitudes will
    be within a few percent of each other at small n_ext.

    Operates in whatever output space y_obs and y_pred are in (typically log).
    """
    os.makedirs(out_dir, exist_ok=True)
    n_ext = len(y_obs_ext)
    n_tr = len(y_obs_train)
    y_tr_mean = y_obs_train.mean()
    y_ext_mean = y_obs_ext.mean()

    ss_pred = np.sum((y_obs_ext - y_pred_ext) ** 2)
    ss_train_centred_on_ext = np.sum((y_obs_ext - y_tr_mean) ** 2)
    ss_ext_centred = np.sum((y_obs_ext - y_ext_mean) ** 2)
    ss_train_centred = np.sum((y_obs_train - y_tr_mean) ** 2)

    r2_ext = np.corrcoef(y_obs_ext, y_pred_ext)[0, 1] ** 2
    q2_f1 = 1 - ss_pred / ss_train_centred_on_ext
    q2_f2 = 1 - ss_pred / ss_ext_centred
    q2_f3 = 1 - (ss_pred / n_ext) / (ss_train_centred / n_tr)

    out = dict(r2_ext=r2_ext, q2_f1=q2_f1, q2_f2=q2_f2, q2_f3=q2_f3, n_ext=n_ext)
    print(f"[b2] {target_name}: r2_ext={r2_ext:.3f}  Q2_F1={q2_f1:.3f}  Q2_F2={q2_f2:.3f}  Q2_F3={q2_f3:.3f}")

    # Single-target bar chart (call b2 twice to compare both targets)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    _clean_figure(fig)
    labels = [r'$r^{2}_{\,\mathrm{ext}}$', r'$Q^{2}_{\,F1}$',
              r'$Q^{2}_{\,F2}$', r'$Q^{2}_{\,F3}$']
    vals = [r2_ext, q2_f1, q2_f2, q2_f3]
    bars = ax.bar(labels, vals,
                  color=[MATRIX_BLUE, MATRIX_BLUE, MATRIX_BLUE, MATRIX_BLUE],
                  edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Metric value')
    ax.set_title(f'External validation metrics: {target_name}  ($n_{{\\rm ext}}={n_ext}$)')
    #ax.axhline(0.5, color='black', ls='--', lw=0.8, label='0.5 reference')
    # y-axis: bars flush with bottom spine when all positive; expand below 0
    # only if any metric is negative (keeps the bars visible without dead canvas)
    ymin = min(0.0, min(vals) - 0.05) if min(vals) < 0 else 0.0
    ax.set_ylim(ymin, 1.0)
    for b, v in zip(bars, vals):
        # Always place label just above the bar top edge (works for + and -)
        ax.text(b.get_x() + b.get_width()/2,
                max(v, 0) + 0.025,
                f'{v:.3f}', ha='center', fontsize=11, fontweight='bold', color='black')
    # (no legend: bars carry no label series; values are annotated above each bar)
    _clean_axes(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'b2_q2_metrics_{target_name}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# =============================================================================
# (b3) Predictive-variance calibration
# =============================================================================

def run_b3(X_train_base: np.ndarray, y_train_bc: np.ndarray, lam: float,
           feat_names: list,
           gpr_factory: Callable = default_gpr_factory,
           target_name: str = 'target',
           out_dir: str = './analyses_output') -> dict:
    """
    Calibration of GPR pointwise predictive intervals.
    Run LOO predictions with predictive std, then for each nominal confidence level
    measure the empirical fraction of true LOO observations falling inside the interval.
    """
    os.makedirs(out_dir, exist_ok=True)
    add_int = make_add_interactions(feat_names)

    # Get LOO predictions with std in Box-Cox space
    n = len(y_train_bc)
    preds_bc = np.zeros(n)
    stds_bc = np.zeros(n)
    for tr, te in LeaveOneOut().split(X_train_base):
        Xtr = add_int(X_train_base[tr])
        Xte = add_int(X_train_base[te])
        sc = StandardScaler().fit(Xtr)
        gpr = copy.deepcopy(gpr_factory())
        gpr.fit(sc.transform(Xtr), y_train_bc[tr])
        mean, std = gpr.predict(sc.transform(Xte), return_std=True)
        preds_bc[te] = mean.ravel()[0]
        stds_bc[te] = std.ravel()[0]

    # Calibration in Box-Cox space (where the GPR likelihood is Gaussian)
    nominals = np.arange(0.05, 1.0, 0.05)
    empirical = []
    for nom in nominals:
        z = scipy_norm.ppf(0.5 + nom / 2)
        lo = preds_bc - z * stds_bc
        hi = preds_bc + z * stds_bc
        inside = ((y_train_bc >= lo) & (y_train_bc <= hi)).mean()
        empirical.append(inside)
    empirical = np.array(empirical)
    cal_err = float(np.mean(np.abs(empirical - nominals)))
    coverage_95 = float(empirical[np.argmin(np.abs(nominals - 0.95))])

    out = dict(nominals=nominals.tolist(), empirical=empirical.tolist(),
               cal_err=cal_err, coverage_95=coverage_95,
               loo_means_bc=preds_bc.tolist(), loo_stds_bc=stds_bc.tolist())
    print(f"[b3] {target_name}: cal_err(L1)={cal_err:.4f}  95% coverage={coverage_95:.2f}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    _clean_figure(fig)
    ax.plot([0, 1], [0, 1], color='black', ls='--', lw=1.0, label='Perfect calibration')
    ax.plot(nominals, empirical, 'o-', color=MATRIX_BLUE, lw=2,
            markeredgecolor='black', markeredgewidth=0.6, label=f'{target_name}')
    ax.scatter([0.95], [coverage_95], s=130, color=ACCENT_ORANGE,
               edgecolor='black', linewidth=0.8, zorder=5,
               label=f'95% nominal → {coverage_95*100:.1f}% empirical')
    ax.set_xlabel('Nominal confidence level')
    ax.set_ylabel('Empirical coverage (LOO)')
    ax.set_title(f'GPR predictive-variance calibration: {target_name}\n'
                 f'$\\ell_{{1}}$ error = {cal_err:.4f}')
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _clean_axes(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'b3_calibration_{target_name}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# =============================================================================
# (b4) Baseline decomposition
# =============================================================================

def run_b4(X_train_base: np.ndarray, y_train_bc: np.ndarray, y_train_log: np.ndarray,
           lam: float, feat_names: list,
           gpr_factory: Callable = default_gpr_factory,
           target_name: str = 'target',
           out_dir: str = './analyses_output') -> dict:
    """
    Q^2 (LOO) for null / OLS / single-feature (logP, halflife, tortuosity) /
    logP-quintile proxy / full GPR. All baselines trained directly on the
    LOG target (not Box-Cox) so they compete fairly with the GPR in log space.
    The full GPR uses the production Box-Cox pipeline and is back-transformed to log.
    """
    os.makedirs(out_dir, exist_ok=True)
    add_int = make_add_interactions(feat_names)

    y_obs = y_train_log
    y_ref = y_obs.mean()

    results = {'null': 0.0}

    # OLS on 8 base features, trained directly on log target
    preds_ols = np.zeros(len(y_obs))
    for tr, te in LeaveOneOut().split(X_train_base):
        mdl = LinearRegression().fit(X_train_base[tr], y_train_log[tr])
        preds_ols[te] = mdl.predict(X_train_base[te]).ravel()[0]
    results['ols_8feat'] = float(_q2(y_obs, preds_ols, y_ref))

    # Single-feature regressions, on log target
    for single_name, key in [('logP_matrix', 'single_feature_logP'),
                             ('Half life_matrix_avg', 'single_feature_halflife'),
                             ('tortuosity', 'single_feature_tortuosity')]:
        if single_name not in feat_names:
            results[key] = float('nan')
            continue
        col = feat_names.index(single_name)
        preds_s = np.zeros(len(y_obs))
        for tr, te in LeaveOneOut().split(X_train_base):
            x_tr = X_train_base[tr, col:col+1]
            x_te = X_train_base[te, col:col+1]
            mdl = LinearRegression().fit(x_tr, y_train_log[tr])
            preds_s[te] = mdl.predict(x_te).ravel()[0]
        results[key] = float(_q2(y_obs, preds_s, y_ref))

    # logP-quintile proxy (matrix-identity baseline) -- predict per-quintile mean of log target
    col = feat_names.index('logP_matrix')
    logp = X_train_base[:, col]
    q_edges = np.quantile(logp, [0.2, 0.4, 0.6, 0.8])
    quintiles = np.digitize(logp, q_edges)
    preds_q = np.zeros(len(y_obs))
    for i in range(len(y_obs)):
        mask = np.arange(len(y_obs)) != i
        same_bin = (quintiles == quintiles[i]) & mask
        if same_bin.sum() == 0:
            preds_q[i] = y_train_log[mask].mean()
        else:
            preds_q[i] = y_train_log[same_bin].mean()
    results['logP_quintile_proxy'] = float(_q2(y_obs, preds_q, y_ref))

    # Full GPR (Box-Cox + interactions + fold-internal scaler), back-transformed to log
    preds_full = _loo_predict_full(X_train_base, y_train_bc, lam, gpr_factory, add_int)
    results['gpr_full'] = float(_q2(y_obs, preds_full, y_ref))

    print(f"[b4] {target_name} Q^2(LOO) baseline decomposition:")
    for k, v in results.items():
        print(f"    {k:30s}: {v:.3f}")

    labels = ['null', 'OLS (8 feat)', 'single feat\nlogP_matrix',
              'single feat\nhalf-life', 'single feat\ntortuosity',
              'logP-quintile\n(matrix-identity)', 'Full GPR']
    keys = ['null', 'ols_8feat', 'single_feature_logP',
            'single_feature_halflife', 'single_feature_tortuosity',
            'logP_quintile_proxy', 'gpr_full']
    vals = [results[k] for k in keys]

    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    _clean_figure(fig)
    # Colour scheme (two-class feature scheme + neutral baselines + GPR accent):
    #   null, OLS (multivariate), logP-quintile (matrix-identity proxy) → neutral grey
    #     (these are not single-feature baselines, so they have no feature class)
    #   single-feat logP_matrix, single-feat half-life                  → matrix blue
    #   single-feat tortuosity                                          → fibre green
    #   full GPR                                                        → accent red
    colors = [NEUTRAL_GREY, NEUTRAL_GREY,
              MATRIX_BLUE, MATRIX_BLUE, FIBRE_GREEN,
              NEUTRAL_GREY, ACCENT_ORANGE]
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.8)
    ax.axhline(0, color='black', lw=0.9)
    #ax.axhline(0.5, color='black', ls='--', lw=0.8, label='0.5 reference')
    ax.set_ylabel(r'$Q^{2}$ (LOO)', fontsize = 18)
    ax.set_title(f'Baseline decomposition: {target_name}')
    # Expand y-axis so negative bars get visible whitespace below them,
    # which prevents label collisions with the bar bodies near zero
    y_top = max(vals) + 0.10
    y_bot = (min(vals) - 0.10) if min(vals) < 0 else -0.05
    ax.set_ylim(y_bot, y_top)
    # Label placement:
    #   positive bars: text sits just above the bar top
    #   negative bars: text sits well above the zero line (label_pad chosen
    #                  to clear both the bar body and the y=0 axis line)
    y_range = y_top - y_bot
    label_pad_pos = 0.020 * y_range
    label_pad_neg = 0.050 * y_range   # larger offset so label clears the bar
    for b, v in zip(bars, vals):
        if v >= 0:
            y_text = v + label_pad_pos
            va = 'bottom'
        else:
            # Label sits above the zero line with comfortable padding
            y_text = label_pad_neg
            va = 'bottom'
        ax.text(b.get_x() + b.get_width()/2, y_text,
                f'{v:.3f}', ha='center', va=va, fontsize=11, fontweight='bold', color='black')
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    _clean_axes(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'b4_baseline_decomposition_{target_name}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return results


# =============================================================================
# (b5) Bootstrap SHAP stability
# =============================================================================

def run_b5(X_train_base: np.ndarray, y_train_bc: np.ndarray, feat_names: list,
           gpr_factory: Callable = default_gpr_factory,
           n_iter: int = 60, target_name: str = 'target',
           out_dir: str = './analyses_output') -> dict:
    """
    Bootstrap stability of the SHAP top-3 ranking. Resamples the 40-sample
    training set with replacement, refits the GPR with interactions and a
    fold-internal scaler, recomputes mean |SHAP| over the resampled set.

    Note: imports shap lazily.
    """
    import shap
    os.makedirs(out_dir, exist_ok=True)
    add_int = make_add_interactions(feat_names)
    n_feat = len(feat_names)
    n_samples = len(X_train_base)
    rng = np.random.default_rng(RANDOM_SEED)

    top1_counts = {f: 0 for f in feat_names}
    top3_set_counts = {f: 0 for f in feat_names}
    completed = 0

    for it in range(n_iter):
        try:
            idx = rng.integers(0, n_samples, size=n_samples)
            Xb = X_train_base[idx]
            yb = y_train_bc[idx]
            Xb_int = add_int(Xb)
            sc = StandardScaler().fit(Xb_int)
            Xb_s = sc.transform(Xb_int)
            gpr = copy.deepcopy(gpr_factory())
            gpr.fit(Xb_s, yb)

            # SHAP on the 8 base features only (interactions inflate the column count)
            # Use a wrapper that re-applies interactions and scaling
            def f(X_query_base):
                Xq = add_int(X_query_base)
                return gpr.predict(sc.transform(Xq))

            # Background = the bootstrap training set (base features)
            explainer = shap.KernelExplainer(f, Xb[:min(20, len(Xb))], silent=True)
            shap_vals = explainer.shap_values(Xb, nsamples=50, silent=True)
            mean_abs = np.abs(shap_vals).mean(axis=0)

            order = np.argsort(mean_abs)[::-1]
            top1 = feat_names[order[0]]
            top3 = [feat_names[i] for i in order[:3]]
            top1_counts[top1] += 1
            for f_ in top3:
                top3_set_counts[f_] += 1
            completed += 1
            if (it + 1) % 10 == 0:
                print(f"[b5] iter {it+1}/{n_iter}  current top-1: {top1}")
        except Exception as e:
            print(f"[b5] iter {it+1} failed: {e}; continuing")
            continue

    if completed == 0:
        print("[b5] All iterations failed; no results.")
        return {}

    top1_freq = {f: c / completed for f, c in top1_counts.items()}
    top3_freq = {f: c / completed for f, c in top3_set_counts.items()}

    out = dict(n_completed=completed, top1_counts=top1_counts,
               top3_set_counts=top3_set_counts,
               top1_freq=top1_freq, top3_freq=top3_freq)

    print(f"[b5] {target_name}: bootstrap stability over {completed} successful iterations")
    print("    Rank-1 frequency (sorted):")
    for f, v in sorted(top1_freq.items(), key=lambda kv: -kv[1])[:6]:
        print(f"      {f:30s}: {v*100:.1f}%")
    print("    Top-3 set frequency (sorted):")
    for f, v in sorted(top3_freq.items(), key=lambda kv: -kv[1])[:6]:
        print(f"      {f:30s}: {v*100:.1f}%")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    _clean_figure(fig)

    # ----- Top-1 frequency panel (left) -----
    sorted_top1 = sorted(top1_freq.items(), key=lambda kv: -kv[1])
    feats_t1 = [k for k, v in sorted_top1 if v > 0]
    vals_t1 = [top1_freq[k] for k in feats_t1]
    colors_t1 = [_feat_color(f) for f in feats_t1]
    ax[0].barh(feats_t1[::-1],
               [v * 100 for v in vals_t1[::-1]],
               color=colors_t1[::-1],
               edgecolor='black', linewidth=0.7)
    ax[0].set_xlabel('Rank-1 frequency (%)')
    ax[0].set_title(f'Bootstrap rank-1 attribution\n'
                    f'({completed} iterations, {target_name})')
    _clean_axes(ax[0])

    # ----- Top-3 set frequency panel (right) -----
    sorted_top3 = sorted(top3_freq.items(), key=lambda kv: -kv[1])
    feats_t3 = [k for k, v in sorted_top3 if v > 0]
    vals_t3 = [top3_freq[k] for k in feats_t3]
    colors_t3 = [_feat_color(f) for f in feats_t3]
    ax[1].barh(feats_t3[::-1],
               [v * 100 for v in vals_t3[::-1]],
               color=colors_t3[::-1],
               edgecolor='black', linewidth=0.7)
    #ax[1].axvline(50, color='black', ls='--', lw=0.8, label='50%')
    ax[1].set_xlabel('Top-3 set frequency (%)')
    ax[1].set_title(f'Bootstrap top-3 set membership\n'
                    f'({completed} iterations, {target_name})')
    ax[1].legend(loc='upper right', frameon=True, edgecolor='black')
    _clean_axes(ax[1])

    # Shared legend: matrix/fibre colour key (two-class scheme)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=MATRIX_BLUE, edgecolor='black', label='Matrix-derived'),
        Patch(facecolor=FIBRE_GREEN, edgecolor='black', label='Fibre-derived'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=2, frameon=True, edgecolor='black',
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(out_dir, f'b5_bootstrap_stability_{target_name}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# =============================================================================
# (b6) SHAP sensitivity to log transformation
# =============================================================================

def run_b6(X_train_log: np.ndarray, X_train_raw: np.ndarray,
           y_train_bc: np.ndarray, feat_names: list,
           gpr_factory: Callable = default_gpr_factory,
           target_name: str = 'target',
           out_dir: str = './analyses_output') -> dict:
    """
    Compare SHAP mean |value| with and without the log transform on five
    right-skewed features. X_train_log is the matrix as used in the headline
    pipeline (5 features log-transformed); X_train_raw has those 5 features in
    their original (un-log) scale, rest of columns identical.

    Both versions: fit GPR with interactions + fold-internal scaler on the
    full training set; compute SHAP global mean |value| using the training set
    as background.
    """
    import shap
    os.makedirs(out_dir, exist_ok=True)
    add_int = make_add_interactions(feat_names)

    def fit_and_get_shap(X_base):
        Xb_int = add_int(X_base)
        sc = StandardScaler().fit(Xb_int)
        Xs = sc.transform(Xb_int)
        gpr = copy.deepcopy(gpr_factory())
        gpr.fit(Xs, y_train_bc)
        def f(X_query_base):
            Xq = add_int(X_query_base)
            return gpr.predict(sc.transform(Xq))
        explainer = shap.KernelExplainer(f, X_base[:min(20, len(X_base))], silent=True)
        sv = explainer.shap_values(X_base, nsamples=100, silent=True)
        return np.abs(sv).mean(axis=0)

    print("[b6] computing SHAP with log transformation...")
    shap_with_log = fit_and_get_shap(X_train_log)
    print("[b6] computing SHAP without log transformation...")
    shap_without_log = fit_and_get_shap(X_train_raw)

    with_log = {f: float(v) for f, v in zip(feat_names, shap_with_log)}
    no_log = {f: float(v) for f, v in zip(feat_names, shap_without_log)}
    top3_with = [f for f, _ in sorted(with_log.items(), key=lambda kv: -kv[1])[:3]]
    top3_without = [f for f, _ in sorted(no_log.items(), key=lambda kv: -kv[1])[:3]]

    out = dict(shap_with_log=with_log, shap_without_log=no_log,
               top3_with_log=top3_with, top3_without_log=top3_without)

    print(f"[b6] {target_name} mean |SHAP|:")
    print("    feature                       with-log     without-log")
    for f in feat_names:
        print(f"    {f:30s}  {with_log[f]:.3f}       {no_log[f]:.3f}")
    print(f"    Top-3 with log:    {top3_with}")
    print(f"    Top-3 without log: {top3_without}")

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    _clean_figure(fig)
    order = sorted(feat_names, key=lambda f: -(with_log[f] + no_log[f]))
    x = np.arange(len(order))
    w = 0.38
    bar_colors = [_feat_color(f) for f in order]

    # With-log series: solid fill in the feature's class colour
    ax.bar(x - w/2, [with_log[f] for f in order], w,
           color=bar_colors, edgecolor='black', linewidth=0.7)

    # Without-log series: same colour, hatched, lighter (visually distinct
    # but still readable as 'same feature, different transform')
    ax.bar(x + w/2, [no_log[f] for f in order], w,
           color='white', edgecolor='black', linewidth=0.7,
           hatch='///',
           label='Without log transform')
    # Overlay a thin coloured rim so the without-log bars retain feature identity
    for xi, f in zip(x, order):
        ax.bar(xi + w/2, no_log[f], w,
               color='none', edgecolor=_feat_color(f), linewidth=1.6)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_ylabel('Mean $|$SHAP$|$')
    ax.set_title(f'SHAP ranking sensitivity to log-transformation: {target_name}')

    # Two-tier legend: with/without-log + feature-class key
    from matplotlib.patches import Patch
    legend_handles = [
        #Patch(facecolor='lightgrey', edgecolor='black',
              #label='With log transform (headline)'),
        Patch(facecolor='white', edgecolor='black', hatch='///',
              label='Without log transform'),
        Patch(facecolor=MATRIX_BLUE, edgecolor='black', label='Matrix-derived'),
        Patch(facecolor=FIBRE_GREEN, edgecolor='black', label='Fibre-derived'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              frameon=True, edgecolor='black', ncol=1)
    _clean_axes(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'b6_shap_sensitivity_{target_name}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# =============================================================================
# Convenience wrapper
# =============================================================================

def run_all(X_train_base: np.ndarray, y_train_bc: np.ndarray, y_train_log: np.ndarray,
            X_ext_base: np.ndarray, y_ext_log: np.ndarray, ext_pred_log: np.ndarray,
            lam: float, feat_names: list,
            X_train_raw_for_b6: Optional[np.ndarray] = None,
            gpr_factory: Callable = default_gpr_factory,
            n_bootstrap: int = 60,
            target_name: str = 'hixson_rate',
            out_dir: str = './analyses_output') -> dict:
    """
    Run all five analyses. Returns one dict keyed by 'b2', 'b3', 'b4', 'b5', 'b6'.
    Each function also writes a PNG to out_dir.

    If X_train_raw_for_b6 is None, b6 is skipped (it requires the raw, un-log version
    of the 5 right-skewed features).
    """
    all_results = {}
    print(f"\n=== Running all b2-b6 for {target_name} ===\n")

    all_results['b2'] = run_b2(y_ext_log, ext_pred_log, y_train_log,
                               target_name=target_name, out_dir=out_dir)
    all_results['b3'] = run_b3(X_train_base, y_train_bc, lam, feat_names,
                               gpr_factory=gpr_factory, target_name=target_name, out_dir=out_dir)
    all_results['b4'] = run_b4(X_train_base, y_train_bc, y_train_log, lam, feat_names,
                               gpr_factory=gpr_factory, target_name=target_name, out_dir=out_dir)
    all_results['b5'] = run_b5(X_train_base, y_train_bc, feat_names,
                               gpr_factory=gpr_factory, n_iter=n_bootstrap,
                               target_name=target_name, out_dir=out_dir)
    if X_train_raw_for_b6 is not None:
        all_results['b6'] = run_b6(X_train_base, X_train_raw_for_b6, y_train_bc, feat_names,
                                   gpr_factory=gpr_factory, target_name=target_name, out_dir=out_dir)
    else:
        print("[b6] skipped (no X_train_raw_for_b6 provided)")

    print(f"\n=== Done. Figures and results in {out_dir} ===")
    return all_results


# =============================================================================
# Self-contained pipeline (mirrors Robust_GPR_model.ipynb exactly)
# Use this if you want to run analyses_b2_b6.py standalone on dataset.xlsx
# without depending on any other module. Inside your notebook you should
# just pass your own X_train / y_train / etc. directly to run_all().
# =============================================================================

_COLLINEAR_DROP_DEFAULT = [
    'hydrophyllic_gradient', 'matrix_sol_parameter', 'flory_huggins',
    'logKOC_matrix_avg', 'sol_mismatch', 'logP_fiber', 'Lignin_Percentage',
    'Cellulose_Percentage', 'Half life_fiber_avg', 'logKOC_fiber_avg',
    'fiber_CED', 'fiber_rotatable_bond_count', 'Specific Interface Area',
]
_PRELIM_OUTLIER_ROWS = [0, 8, 18]
_N_OUTLIERS = 9
_N_EXT = 9


def _kennard_stone(X, n_select):
    """Kennard-Stone external-test selection on standardised X."""
    from scipy.spatial.distance import cdist
    dist = cdist(X, X, 'euclidean')
    i1, i2 = np.unravel_index(np.argmax(dist), dist.shape)
    sel = [int(i1), int(i2)]
    rem = list(set(range(len(X))) - set(sel))
    while len(sel) < n_select:
        d = cdist(X[rem], X[sel], 'euclidean')
        idx = int(np.argmax(np.min(d, axis=1)))
        sel.append(rem[idx])
        rem.remove(rem[idx])
    return sel


def reproduce_pipeline(dataset_path: str,
                       target: str = 'hixson_rate',
                       collinear_drop: Optional[list] = None,
                       verbose: bool = True) -> dict:
    """
    Reproduce the full Robust_GPR_model.ipynb pipeline starting from dataset.xlsx
    and return all variables needed by run_all().

    Returns a dict with keys:
        X_train_base, y_train_bc, y_train_log, X_ext_base, y_ext_log,
        ext_pred_log, lam, feat_names, X_train_raw_for_b6
    """
    if collinear_drop is None:
        collinear_drop = _COLLINEAR_DROP_DEFAULT

    # 1. Load and rename
    df = pd.read_excel(dataset_path)
    TARGET_FO, TARGET_HIX = 'first_order_rate', 'hixson_rate'

    # 2. Drop NA across feature columns
    all_feats = [c for c in df.columns if c not in (TARGET_FO, TARGET_HIX)]
    df = df.dropna(subset=all_feats).reset_index(drop=True)

    # 3. Log-transform the 5 right-skewed features (raw values preserved separately)
    df_raw = df.copy()
    for f in LOG_TRANSFORM_FEATS:
        if f in df.columns and (df[f].values > 0).all():
            df[f] = np.log(df[f].values)

    # 4. Drop collinear features
    df.drop(columns=[c for c in collinear_drop if c in df.columns], inplace=True)
    df_raw.drop(columns=[c for c in collinear_drop if c in df_raw.columns], inplace=True)
    feats_model = [f for f in all_feats if f in df.columns]

    # 5. Drop preliminary outliers (rows 0, 8, 18 of the cleaned table)
    mask = np.ones(len(df), dtype=bool)
    mask[_PRELIM_OUTLIER_ROWS] = False
    df = df[mask].reset_index(drop=True)
    df_raw = df_raw[mask].reset_index(drop=True)

    # 6. Kennard-Stone external split (n=9)
    X_all = df[feats_model].values.astype(float)
    X_all_raw = df_raw[feats_model].values.astype(float)
    raw_yfo  = df[TARGET_FO].values
    raw_yhix = df[TARGET_HIX].values
    log_yfo  = np.log(raw_yfo)
    log_yhix = np.log(raw_yhix)

    sc_ks = StandardScaler()
    ks_idx = _kennard_stone(sc_ks.fit_transform(X_all), n_select=_N_EXT)
    ks_mask = np.zeros(len(X_all), dtype=bool)
    ks_mask[ks_idx] = True

    X_train_pool = X_all[~ks_mask]
    X_test_full = X_all[ks_mask]
    X_train_pool_raw = X_all_raw[~ks_mask]
    raw_yfo_pool = raw_yfo[~ks_mask]
    raw_yhix_pool = raw_yhix[~ks_mask]
    log_yfo_pool = log_yfo[~ks_mask]
    log_yhix_pool = log_yhix[~ks_mask]
    log_yfo_te = log_yfo[ks_mask]
    log_yhix_te = log_yhix[ks_mask]

    # 7. Identify N=9 outliers in training pool via Box-Cox first-order residuals
    bc_yfo_pool, lam_fo_pool = boxcox(raw_yfo_pool)
    add_int = make_add_interactions(feats_model)
    gpr_id = GaussianProcessRegressor(
        kernel=C(1.0,(1e-2,1e2)) * RBF(1.0,(0.5,20.0)) + WhiteKernel(0.1,(1e-2,1.0)),
        alpha=0.1, n_restarts_optimizer=5, normalize_y=True, random_state=RANDOM_SEED)
    preds_id_bc = np.zeros(len(log_yfo_pool))
    for tr, te in LeaveOneOut().split(X_train_pool):
        Xtr = add_int(X_train_pool[tr])
        Xte = add_int(X_train_pool[te])
        sc = StandardScaler()
        m = copy.deepcopy(gpr_id)
        m.fit(sc.fit_transform(Xtr), bc_yfo_pool[tr])
        preds_id_bc[te] = m.predict(sc.transform(Xte)).ravel()[0]
    preds_id_log = bc_to_log(preds_id_bc, lam_fo_pool)
    residuals = log_yfo_pool - preds_id_log
    std_res = residuals / residuals.std()
    outlier_idx = np.argsort(np.abs(std_res))[-_N_OUTLIERS:].tolist()

    keep = np.ones(len(X_train_pool), dtype=bool)
    keep[outlier_idx] = False
    X_train = X_train_pool[keep]
    X_train_raw = X_train_pool_raw[keep]
    raw_yfo_tr = raw_yfo_pool[keep]
    raw_yhix_tr = raw_yhix_pool[keep]
    log_yfo_tr = log_yfo_pool[keep]
    log_yhix_tr = log_yhix_pool[keep]

    # 8. Box-Cox on the final 40 training targets, separately per kinetic endpoint
    bc_yfo_tr,  lam_fo  = boxcox(raw_yfo_tr)
    bc_yhix_tr, lam_hix = boxcox(raw_yhix_tr)

    # 9. Fit final GPR on the chosen target and get external predictions
    if target == TARGET_FO:
        bc_y_tr, lam, log_y_tr, log_y_te = bc_yfo_tr, lam_fo, log_yfo_tr, log_yfo_te
    elif target == TARGET_HIX:
        bc_y_tr, lam, log_y_tr, log_y_te = bc_yhix_tr, lam_hix, log_yhix_tr, log_yhix_te
    else:
        raise ValueError(f"target must be {TARGET_FO!r} or {TARGET_HIX!r}, got {target!r}")

    gpr_final = default_gpr_factory(n_restarts=10)
    Xtr_int = add_int(X_train)
    Xte_int = add_int(X_test_full)
    sc = StandardScaler().fit(Xtr_int)
    gpr_final.fit(sc.transform(Xtr_int), bc_y_tr)
    ext_pred_bc = gpr_final.predict(sc.transform(Xte_int))
    ext_pred_log = bc_to_log(ext_pred_bc, lam)

    if verbose:
        print(f"  Reconstructed pipeline for {target}")
        print(f"    Training set: {X_train.shape[0]} samples x {X_train.shape[1]} features")
        print(f"    External test: {X_test_full.shape[0]} samples")
        print(f"    Box-Cox lambda = {lam:.4f}")
        print(f"    Final fitted kernel: {gpr_final.kernel_}")

    return dict(
        X_train_base=X_train,
        y_train_bc=bc_y_tr,
        y_train_log=log_y_tr,
        X_ext_base=X_test_full,
        y_ext_log=log_y_te,
        ext_pred_log=ext_pred_log,
        lam=lam,
        feat_names=feats_model,
        X_train_raw_for_b6=X_train_raw,
    )


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == '__main__':
    """
    Standalone usage:
        python analyses_b2_b6.py /path/to/dataset.xlsx [target] [out_dir]

    target:  'hixson_rate' (default) or 'first_order_rate'
    out_dir: './analyses_output' by default

    Or, from inside Robust_GPR_model.ipynb, just import and call run_all() with
    your own variables -- see the docstring at the top of the file.
    """
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo dataset path provided. Usage:")
        print("    python analyses_b2_b6.py /path/to/dataset.xlsx [target] [out_dir]")
        sys.exit(0)

    dataset_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else 'hixson_rate'
    out_dir = sys.argv[3] if len(sys.argv) > 3 else './analyses_output'

    if not os.path.exists(dataset_path):
        print(f"ERROR: dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Reproducing pipeline from {dataset_path} for target={target}...")
    pipe = reproduce_pipeline(dataset_path, target=target, verbose=True)

    print(f"\nRunning b2-b6 analyses, writing to {out_dir}/")
    results = run_all(
        target_name=target,
        out_dir=out_dir,
        n_bootstrap=60,
        **pipe,
    )
    print("\nDone.")

