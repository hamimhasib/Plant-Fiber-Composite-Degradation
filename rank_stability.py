#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_stability.py
===================
Companion to leakage_corrected_validation.py. Tests whether the *governing-factor
ranking* (not the point predictions) is stable when whole studies are withheld.

Rationale: the model's rate predictions do not transfer across unseen studies
(see leakage_corrected_validation.py). This script asks the separate question of
whether the FACTORS the model identifies as governing soil degradation persist
study-to-study. For each leave-one-study-out (LOSO) fold the GPR is refit and the
permutation importance of the eight base descriptors is recomputed; the stability
of the resulting ranking is then summarised.

Produces the values reported in Supplementary Note S10.5 / Table S15:
    * reference (all-studies) importance ranking of the 8 base descriptors,
    * per-descriptor frequency of being ranked top-1 and top-3 across LOSO folds,
    * Kendall tau of each fold's ranking vs the all-studies ranking,
    * exact-recovery rate of the reference top-3 set.

Inputs (same as the deposit): dataset.xlsx, applicability_domain.csv
Run:   python factor_stability.py            # prints the table; --csv also writes it
"""

from __future__ import annotations
import argparse
import copy
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import boxcox, kendalltau
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
TARGET = "first_order_rate"          # primary target; set to "hixson_rate" to repeat
N_PERM = 8                           # permutation repeats per feature per fold

LOG_FEATURES = ["Half life_matrix_avg", "tortuosity", "interface_rot",
                "matrix_CED", "fiber_dia_or_average_particle_size_mm"]
COLLINEAR_DROP = ["hydrophyllic_gradient", "matrix_sol_parameter", "flory_huggins",
                  "logKOC_matrix_avg", "sol_mismatch", "logP_fiber", "Lignin_Percentage",
                  "Cellulose_Percentage", "Half life_fiber_avg", "logKOC_fiber_avg",
                  "fiber_CED", "fiber_rotatable_bond_count", "Specific Interface Area"]
PRELIM_OUTLIER_ROWS = [0, 8, 18]
MATRIX_FEATS = ["Half life_matrix_avg", "logP_matrix", "matrix_CED"]
FIBER_FEATS = ["Hemicellulose_Percentage", "tortuosity", "interface_rot",
               "fiber_sol_parameter", "fiber_dia_or_average_particle_size_mm"]
PRETTY = {"Half life_matrix_avg": "matrix half-life",
          "logP_matrix": "matrix logP (hydrophobicity)",
          "matrix_CED": "matrix CED",
          "Hemicellulose_Percentage": "hemicellulose %",
          "tortuosity": "tortuosity",
          "interface_rot": "interface_rot",
          "fiber_sol_parameter": "fibre solubility parameter",
          "fiber_dia_or_average_particle_size_mm": "fibre diameter"}


def _normalise_ref(ref: str) -> str:
    s = re.sub(r"\d+$", "", str(ref).strip())
    s = re.sub(r"[().,]", " ", s)
    s = re.sub(r"\bel\b", "et", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_frame(data_dir: Path):
    df = pd.read_excel(data_dir / "dataset.xlsx")
    ad = pd.read_csv(data_dir / "applicability_domain.csv")
    if not np.allclose(df["first_order_rate"].values,
                       ad["Calculated First Order Rate"].values, rtol=1e-3):
        raise ValueError("dataset.xlsx and applicability_domain.csv are not row-aligned")
    df["study"] = ad["Ref."].map(_normalise_ref).values

    targets = ["first_order_rate", "hixson_rate"]
    all_feats = [c for c in df.columns if c not in targets + ["study"]]
    frame = df.dropna(subset=all_feats).reset_index(drop=True)
    for f in LOG_FEATURES:
        if (frame[f].values > 0).all():
            frame[f] = np.log(frame[f].values)
    frame = frame.drop(columns=COLLINEAR_DROP)
    feats = [f for f in all_feats if f in frame.columns]
    keep = np.ones(len(frame), bool)
    keep[PRELIM_OUTLIER_ROWS] = False
    return frame[keep].reset_index(drop=True), feats


def make_gpr():
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (0.5, 20.0))
              + WhiteKernel(0.1, (1e-2, 1.0)))
    return GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=4,
                                    normalize_y=True, random_state=SEED)


def add_interactions(X, m_idx, f_idx):
    return np.hstack([X] + [X[:, m:m + 1] * X[:, f:f + 1] for m in m_idx for f in f_idx])


def permutation_importance(model, scaler, X_tr, y_tr, m_idx, f_idx, rng):
    """Drop in training R^2 when each base descriptor is shuffled (mean over N_PERM)."""
    base = model.score(scaler.transform(add_interactions(X_tr, m_idx, f_idx)), y_tr)
    imp = np.zeros(X_tr.shape[1])
    for c in range(X_tr.shape[1]):
        drops = []
        for _ in range(N_PERM):
            Xp = X_tr.copy()
            Xp[:, c] = rng.permutation(Xp[:, c])
            drops.append(base - model.score(scaler.transform(add_interactions(Xp, m_idx, f_idx)), y_tr))
        imp[c] = float(np.mean(drops))
    return imp


def run(frame, feats):
    X = frame[feats].values
    groups = frame["study"].values
    raw = frame[TARGET].values
    m_idx = [feats.index(f) for f in MATRIX_FEATS]
    f_idx = [feats.index(f) for f in FIBER_FEATS]
    rng = np.random.default_rng(0)

    # reference ranking on all studies
    bc, _ = boxcox(raw)
    sc = StandardScaler()
    mdl = copy.deepcopy(make_gpr())
    mdl.fit(sc.fit_transform(add_interactions(X, m_idx, f_idx)), bc)
    ref_imp = permutation_importance(mdl, sc, X, bc, m_idx, f_idx, rng)
    ref_rank = list(np.argsort(ref_imp)[::-1])
    ref_top3 = set(ref_rank[:3])

    # per-fold rankings
    top1, top3_sets, fold_ranks = [], [], []
    for tr, _te in LeaveOneGroupOut().split(X, groups=groups):
        bc, _ = boxcox(raw[tr])
        sc = StandardScaler()
        mdl = copy.deepcopy(make_gpr())
        mdl.fit(sc.fit_transform(add_interactions(X[tr], m_idx, f_idx)), bc)
        imp = permutation_importance(mdl, sc, X[tr], bc, m_idx, f_idx, rng)
        order = list(np.argsort(imp)[::-1])
        top1.append(order[0]); top3_sets.append(set(order[:3])); fold_ranks.append(order)

    n = len(fold_ranks)
    table = pd.DataFrame([{
        "descriptor": PRETTY[feats[c]],
        "reference_importance": round(float(ref_imp[c]), 3),
        "reference_rank": ref_rank.index(c) + 1,
        "top1_freq_%": round(100 * sum(o == c for o in top1) / n, 0),
        "top3_freq_%": round(100 * sum(c in s for s in top3_sets) / n, 0),
    } for c in ref_rank])

    taus = [kendalltau(ref_rank, o).correlation for o in fold_ranks]
    overlaps = [len(ref_top3 & s) for s in top3_sets]
    summary = {
        "n_folds": n,
        "kendall_tau_mean": round(float(np.mean(taus)), 2),
        "kendall_tau_min": round(float(np.min(taus)), 2),
        "top3_mean_overlap_of_3": round(float(np.mean(overlaps)), 1),
        "top3_exact_recovery_%": round(100 * sum(o == 3 for o in overlaps) / n, 0),
    }
    return table, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    frame, feats = load_frame(args.data_dir)
    table, summary = run(frame, feats)
    print(f"Governing-factor stability over {summary['n_folds']} leave-one-study-out folds "
          f"(target = {TARGET})\n")
    print(table.to_string(index=False))
    print("\nRanking stability vs the all-studies ranking:")
    print(f"  Kendall tau: mean {summary['kendall_tau_mean']}, min {summary['kendall_tau_min']} "
          f"(1 = identical order)")
    print(f"  Reference top-3 set recovered: mean {summary['top3_mean_overlap_of_3']}/3, "
          f"exact in {summary['top3_exact_recovery_%']:.0f}% of folds")

    if args.csv:
        table.to_csv("table_S15_factor_stability.csv", index=False)
        pd.DataFrame([summary]).to_csv("factor_stability_summary.csv", index=False)
        print("\nWrote: table_S15_factor_stability.csv, factor_stability_summary.csv")


if __name__ == "__main__":
    main()
