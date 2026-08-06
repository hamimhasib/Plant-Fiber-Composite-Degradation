#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leakage_corrected_validation.py
================================
Reviewer-response / revision analyses for the plant-fibre-composite soil-degradation
QSPR. Reproduces the grouped-validation and applicability-domain results reported in
the revised manuscript and Supplementary Information:

    * Table S13  - leave-one-out (LOO) vs leave-one-STUDY-out (LOSO) cross-validation,
                   on the full modelling set and restricted to the applicability domain,
                   for both kinetic targets.
    * Baseline   - null / matrix-identity-lookup / full-GPR under LOSO (in-domain),
                   quantifying the descriptor block's incremental out-of-study value.
    * Note S6.1  - study overlap of the Kennard-Stone external set with the training pool.
    * Table S14  - the nine largest-residual composites and their applicability-domain status.
    * (support)  - association of matrix logP with rate within genuinely biodegradable
                   matrices only (supports the mass-loss/endpoint discussion).

The GPR pipeline (8 base descriptors after collinearity pruning + 15 matrix x fibre
interaction terms, Box-Cox target, RBF + White kernel) is identical to GPR_prediction.ipynb;
only the cross-validation *grouping* and the reporting differ.

Inputs (place in the same directory, as in the Zenodo deposit):
    dataset.xlsx                 - 62 composites x 23 columns (21 descriptors + 2 targets)
    applicability_domain.csv     - per-composite Sample Name, Matrix, Filler, Ref., Inside AD?

Run:
    pip install pandas numpy scipy scikit-learn openpyxl
    python leakage_corrected_validation.py            # prints tables; --csv also writes them

Author: (add author)   License: (match repository)
"""

from __future__ import annotations
import argparse
import copy
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import boxcox, spearmanr, pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration - mirrors GPR_prediction.ipynb                                 #
# --------------------------------------------------------------------------- #
SEED = 42
TARGETS = ["first_order_rate", "hixson_rate"]

LOG_FEATURES = [
    "Half life_matrix_avg", "tortuosity", "interface_rot",
    "matrix_CED", "fiber_dia_or_average_particle_size_mm",
]
COLLINEAR_DROP = [
    "hydrophyllic_gradient", "matrix_sol_parameter", "flory_huggins",
    "logKOC_matrix_avg", "sol_mismatch", "logP_fiber", "Lignin_Percentage",
    "Cellulose_Percentage", "Half life_fiber_avg", "logKOC_fiber_avg",
    "fiber_CED", "fiber_rotatable_bond_count", "Specific Interface Area",
]
PRELIM_OUTLIER_ROWS = [0, 8, 18]          # Mahalanobis outliers removed before modelling
MATRIX_FEATS = ["Half life_matrix_avg", "logP_matrix", "matrix_CED"]
FIBER_FEATS = [
    "Hemicellulose_Percentage", "tortuosity", "interface_rot",
    "fiber_sol_parameter", "fiber_dia_or_average_particle_size_mm",
]

# Matrices that genuinely biodegrade (used only for the supporting logP analysis).
BIODEGRADABLE_MATRICES = {
    "Starch", "MaterBi-Y", "TPS", "PVA/Starch/Glycerol", "PHA", "PLA/TPS", "PLA",
    "PHBV", "Bio-PBS", "PBS", "PBSu", "PCL", "PBSA", "poly(3HB-co-3HHx)",
}


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
def _normalise_ref(ref: str) -> str:
    """Collapse a free-text reference into a study key for grouping."""
    s = str(ref)
    s = re.sub(r"\d+$", "", s.strip())        # trailing footnote digits
    s = re.sub(r"[().,]", " ", s)
    s = re.sub(r"\bel\b", "et", s)            # 'el al' typo -> 'et al'
    return re.sub(r"\s+", " ", s).strip().lower()


def load_dataset(data_dir: Path):
    """Load descriptors + targets and attach study / matrix / AD / biodegradable labels."""
    df = pd.read_excel(data_dir / "dataset.xlsx")
    ad = pd.read_csv(data_dir / "applicability_domain.csv")

    # the two files are row-aligned; verify via the shared rate column
    if not np.allclose(df["first_order_rate"].values,
                       ad["Calculated First Order Rate"].values, rtol=1e-3):
        raise ValueError("dataset.xlsx and applicability_domain.csv are not row-aligned")

    df["study"] = ad["Ref."].map(_normalise_ref).values
    df["Matrix"] = ad["Matrix"].values
    df["in_AD"] = ad["Inside AD?"].astype(str).str.startswith("Yes").values
    df["is_biodegradable"] = df["Matrix"].isin(BIODEGRADABLE_MATRICES).values
    return df


def build_modelling_frame(df: pd.DataFrame):
    """Apply the notebook's cleaning: dropna -> log-transform -> collinearity drop ->
    remove the 3 preliminary (Mahalanobis) outliers. Returns (frame, feature_list)."""
    label_cols = ["study", "Matrix", "in_AD", "is_biodegradable"]
    all_feats = [c for c in df.columns if c not in TARGETS + label_cols]

    frame = df.dropna(subset=all_feats).reset_index(drop=True).copy()
    for f in LOG_FEATURES:
        if (frame[f].values > 0).all():
            frame[f] = np.log(frame[f].values)
    frame = frame.drop(columns=COLLINEAR_DROP)
    feats = [f for f in all_feats if f in frame.columns]

    keep = np.ones(len(frame), dtype=bool)
    keep[PRELIM_OUTLIER_ROWS] = False
    frame = frame[keep].reset_index(drop=True)
    return frame, feats


# --------------------------------------------------------------------------- #
# Model pipeline (identical to the published model)                            #
# --------------------------------------------------------------------------- #
def make_gpr() -> GaussianProcessRegressor:
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (0.5, 20.0))
              + WhiteKernel(0.1, (1e-2, 1.0)))
    return GaussianProcessRegressor(kernel=kernel, alpha=0.1,
                                    n_restarts_optimizer=5, normalize_y=True,
                                    random_state=SEED)


def _interaction_indices(feats):
    m = [feats.index(f) for f in MATRIX_FEATS]
    f = [feats.index(f) for f in FIBER_FEATS]
    return m, f


def add_interactions(X, m_idx, f_idx):
    """Append the 15 matrix x fibre product terms to the 8 base descriptors."""
    return np.hstack([X] + [X[:, m:m + 1] * X[:, f:f + 1] for m in m_idx for f in f_idx])


def _boxcox_inverse_to_log(pred, lam, eps=1e-10):
    if abs(lam) < 1e-2:
        return pred
    return np.log(np.clip(np.power(np.clip(pred * lam + 1.0, eps, None), 1.0 / lam), eps, None))


def cross_validate(X_base, raw_y, splitter, m_idx, f_idx, groups=None):
    """Generic CV in the log-rate scoring space. Box-Cox is fit within each training
    fold (no leakage of the target transform). Returns (log_y_true, log_y_pred)."""
    log_y = np.log(raw_y)
    pred = np.full(len(raw_y), np.nan)
    iterator = (splitter.split(X_base, groups=groups) if groups is not None
                else splitter.split(X_base))
    for tr, te in iterator:
        bc, lam = boxcox(raw_y[tr])
        scaler = StandardScaler()
        model = copy.deepcopy(make_gpr())
        model.fit(scaler.fit_transform(add_interactions(X_base[tr], m_idx, f_idx)), bc)
        raw_pred = model.predict(scaler.transform(add_interactions(X_base[te], m_idx, f_idx)))
        pred[te] = _boxcox_inverse_to_log(raw_pred, lam)
    return log_y, pred


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def ccc(y, yhat):
    """Lin's concordance correlation coefficient."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    r = np.corrcoef(y, yhat)[0, 1]
    return (2 * r * y.std() * yhat.std()
            / (y.std() ** 2 + yhat.std() ** 2 + (y.mean() - yhat.mean()) ** 2))


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _score(y, yhat, mask=None):
    y, yhat = np.asarray(y), np.asarray(yhat)
    if mask is not None:
        y, yhat = y[mask], yhat[mask]
    return dict(CCC=ccc(y, yhat), Q2=r2(y, yhat),
                Spearman=spearmanr(y, yhat).correlation, RMSE=rmse(y, yhat), n=len(y))


# --------------------------------------------------------------------------- #
# Analyses                                                                     #
# --------------------------------------------------------------------------- #
def table_s13_validation(frame, feats):
    """LOO vs LOSO (whole set and in-domain), both targets."""
    X = frame[feats].values
    groups = frame["study"].values
    in_ad = frame["in_AD"].values
    m_idx, f_idx = _interaction_indices(feats)

    rows = []
    for tgt in TARGETS:
        raw = frame[tgt].values
        ly, p_loo = cross_validate(X, raw, LeaveOneOut(), m_idx, f_idx)
        ly, p_lso = cross_validate(X, raw, LeaveOneGroupOut(), m_idx, f_idx, groups=groups)
        for label, pred, mask in [
            ("LOO (composite-level)", p_loo, None),
            ("LOSO (study-level)", p_lso, None),
            ("LOSO, in-domain", p_lso, in_ad),
        ]:
            s = _score(ly, pred, mask)
            rows.append({"target": tgt, "scheme": label, **{k: round(v, 3) for k, v in s.items()}})
    return pd.DataFrame(rows)


def baseline_decomposition(frame, feats, target="first_order_rate"):
    """null / matrix-identity lookup / full GPR under LOSO, scored in-domain."""
    X = frame[feats].values
    groups = frame["study"].values
    in_ad = frame["in_AD"].values
    matrices = frame["Matrix"].values
    raw = frame[target].values
    log_y = np.log(raw)
    m_idx, f_idx = _interaction_indices(feats)

    def loso_predictions(kind):
        pred = np.full(len(raw), np.nan)
        for tr, te in LeaveOneGroupOut().split(X, groups=groups):
            if kind == "null":
                pred[te] = log_y[tr].mean()
            elif kind == "matrix":                       # predict mean rate of held-out matrix class
                global_mean = log_y[tr].mean()
                for j, idx in enumerate(te):
                    same = log_y[tr][matrices[tr] == matrices[te][j]]
                    pred[idx] = same.mean() if len(same) else global_mean
            else:                                        # full GPR
                bc, lam = boxcox(raw[tr])
                sc = StandardScaler()
                mdl = copy.deepcopy(make_gpr())
                mdl.fit(sc.fit_transform(add_interactions(X[tr], m_idx, f_idx)), bc)
                rp = mdl.predict(sc.transform(add_interactions(X[te], m_idx, f_idx)))
                pred[te] = _boxcox_inverse_to_log(rp, lam)
        return pred

    rows = []
    for kind, label in [("null", "global-mean (null)"),
                        ("matrix", "matrix-identity lookup"),
                        ("gpr", "full GPR (8 base + 15 interactions)")]:
        s = _score(log_y, loso_predictions(kind), in_ad)
        rows.append({"model": label, "R2_in_domain": round(s["Q2"], 3),
                     "CCC_in_domain": round(s["CCC"], 3), "n": s["n"]})
    return pd.DataFrame(rows)


def _kennard_stone(X_scaled, n):
    d = cdist(X_scaled, X_scaled)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    selected = [i, j]
    remaining = list(set(range(len(X_scaled))) - set(selected))
    while len(selected) < n:
        dd = cdist(X_scaled[remaining], X_scaled[selected])
        b = int(np.argmax(np.min(dd, axis=1)))
        selected.append(remaining[b])
        remaining.pop(b)
    return np.array(selected)


def external_set_overlap(frame, feats, n_external=9):
    """How many Kennard-Stone external composites share a study with the training pool."""
    X = frame[feats].values
    groups = frame["study"].values
    ext = _kennard_stone(StandardScaler().fit_transform(X), n_external)
    train_studies = {groups[i] for i in range(len(X)) if i not in ext}
    leaky = int(sum(1 for i in ext if groups[i] in train_studies))
    return leaky, n_external


def table_s14_removed_outliers(frame, feats, target="first_order_rate", n_remove=9):
    """The n largest-residual composites and their applicability-domain status."""
    X = frame[feats].values
    in_ad = frame["in_AD"].values
    matrices = frame["Matrix"].values
    studies = frame["study"].values
    raw = frame[target].values
    log_y = np.log(raw)
    m_idx, f_idx = _interaction_indices(feats)

    pred = np.full(len(raw), np.nan)
    for tr, te in LeaveOneOut().split(X):
        bc, lam = boxcox(raw[tr])
        sc = StandardScaler()
        mdl = copy.deepcopy(make_gpr())
        mdl.fit(sc.fit_transform(add_interactions(X[tr], m_idx, f_idx)), bc)
        rp = mdl.predict(sc.transform(add_interactions(X[te], m_idx, f_idx)))
        pred[te] = _boxcox_inverse_to_log(rp, lam)

    z = (log_y - pred) / (log_y - pred).std()
    idx = np.argsort(np.abs(z))[-n_remove:][::-1]
    rows = [{"rank": r + 1, "matrix": matrices[i], "study": studies[i],
             "abs_std_residual": round(float(abs(z[i])), 2),
             "in_applicability_domain": bool(in_ad[i])}
            for r, i in enumerate(idx)]
    out = pd.DataFrame(rows)
    out.attrs["n_outside_domain"] = int((~out["in_applicability_domain"]).sum())
    return out


def logp_within_biodegradable(frame, target="first_order_rate"):
    """Association of matrix logP with log(rate): full set vs biodegradable-only."""
    log_rate = np.log(frame[target].values)
    logp = frame["logP_matrix"].values
    bd = frame["is_biodegradable"].values.astype(bool)

    def block(mask):
        pr = pearsonr(logp[mask], log_rate[mask])
        sp = spearmanr(logp[mask], log_rate[mask])
        return dict(n=int(mask.sum()),
                    pearson_r=round(pr[0], 3), pearson_p=round(pr[1], 3),
                    spearman=round(sp[0], 3), spearman_p=round(sp[1], 3))

    return pd.DataFrame([
        {"subset": "full set", **block(np.ones(len(bd), bool))},
        {"subset": "biodegradable matrices only", **block(bd)},
    ])


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("."),
                    help="directory containing dataset.xlsx and applicability_domain.csv")
    ap.add_argument("--csv", action="store_true", help="also write the tables as CSV files")
    args = ap.parse_args()

    df = load_dataset(args.data_dir)
    frame, feats = build_modelling_frame(df)
    print(f"Modelling set: {len(frame)} composites | {frame['study'].nunique()} studies | "
          f"{int(frame['in_AD'].sum())} in-domain | {int(frame['is_biodegradable'].sum())} biodegradable\n")

    s13 = table_s13_validation(frame, feats)
    print("=== Table S13 - LOO vs LOSO cross-validation (log-rate scale) ===")
    print(s13.to_string(index=False), "\n")

    base = baseline_decomposition(frame, feats)
    print("=== Baseline decomposition under LOSO, in-domain (first_order_rate) ===")
    print(base.to_string(index=False), "\n")

    leaky, n_ext = external_set_overlap(frame, feats)
    print(f"=== Kennard-Stone external set independence ===")
    print(f"{leaky} of {n_ext} external composites share a study with the training pool\n")

    s14 = table_s14_removed_outliers(frame, feats)
    print("=== Table S14 - largest-residual composites and domain status ===")
    print(s14.to_string(index=False))
    print(f"{s14.attrs['n_outside_domain']} of {len(s14)} fall outside the applicability domain\n")

    logp = logp_within_biodegradable(frame)
    print("=== Supporting: logP_matrix vs log(rate), full set vs biodegradable-only ===")
    print(logp.to_string(index=False), "\n")

    if args.csv:
        s13.to_csv("table_S13_grouped_validation.csv", index=False)
        base.to_csv("baseline_decomposition_loso.csv", index=False)
        s14.to_csv("table_S14_removed_outliers.csv", index=False)
        logp.to_csv("logp_within_biodegradable.csv", index=False)
        print("Wrote: table_S13_grouped_validation.csv, baseline_decomposition_loso.csv, "
              "table_S14_removed_outliers.csv, logp_within_biodegradable.csv")


if __name__ == "__main__":
    main()
