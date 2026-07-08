#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial-distress constrained CQP calculations used in Example~5 of the
manuscript.

The script implements the nonnegative-spread constrained estimators used for
Table 2.  Model M2 is solved by enumeration of fixed slope-sign branches. Model
M3 is solved with the signed-input spread equations because the financial ratios
may be negative.

Outputs
-------
outputs/example1_constrained_m2_coefficients.csv
outputs/example1_constrained_m2_metrics.csv
outputs/example1_constrained_m3_coefficients.csv
outputs/example1_constrained_m3_metrics.csv
outputs/example1_table2_revised_constrained.csv

Dependencies
------------
numpy, pandas, scipy
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import itertools
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.optimize import lsq_linear

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)

FEATURES = ["x1", "x2", "x3"]
N_FEATURES = len(FEATURES)


@dataclass
class Example1Data:
    X_modal: np.ndarray  # crisp/modal predictors, shape (n, 3)
    X_left: np.ndarray   # left spreads of fuzzy predictors, shape (n, 3)
    X_right: np.ndarray  # right spreads of fuzzy predictors, shape (n, 3)
    Y: np.ndarray        # triangular fuzzy response in modal-spread form, shape (n, 3)
    feature_names: list[str]


def load_example1_data() -> Example1Data:
    """Return the financial-distress data in the coding used by the manuscript.

    The displayed input triples are written as triangular vertices
    ``<lower endpoint, modal value, upper endpoint>_T``.  In the calculations
    they are converted to modal-spread form ``(modal, modal-lower,
    upper-modal)_T``.

    The binary labels are converted to fuzzy distress scores as stated in the
    manuscript:

    * category 0 -> ``(0.98, 0.08, 0.01)_T`` (high distress),
    * category 1 -> ``(0.02, 0.01, 0.08)_T`` (low distress).
    """
    rows = [
        (0, (6.21, 7.20, 7.93), (-24.70, -22.60, -19.46), (1.56, 2.00, 2.37)),
        (0, (-4.20, -4.00, -3.62), (-16.83, -15.80, -14.69), (1.63, 2.10, 2.49)),
        (0, (-50.94, -48.20, -44.56), (5.81, 6.80, 7.36), (1.24, 1.60, 1.90)),
        (0, (-31.17, -27.90, -24.57), (5.85, 6.30, 7.11), (1.01, 1.30, 1.54)),
        (0, (-40.04, -38.00, -35.43), (1.51, 1.60, 1.70), (0.93, 1.20, 1.42)),
        (0, (18.49, 20.80, 23.46), (-4.68, -4.30, -3.79), (0.78, 1.00, 1.19)),
        (0, (2.90, 3.30, 3.68), (-3.91, -3.50, -3.07), (0.86, 1.10, 1.31)),
        (0, (-20.42, -18.10, -15.97), (-6.93, -6.50, -5.61), (0.70, 0.90, 1.07)),
        (0, (-23.28, -20.30, -18.21), (-19.04, -17.40, -15.33), (0.78, 1.00, 1.19)),
        (0, (-12.81, -11.40, -10.55), (4.53, 4.80, 5.45), (0.70, 0.90, 1.07)),
        (0, (-9.37, -8.80, -8.06), (-10.01, -9.10, -8.47), (0.70, 0.90, 1.07)),
        (0, (-14.78, -13.10, -11.83), (-19.04, -17.60, -15.78), (0.70, 0.90, 1.07)),
        (0, (-19.58, -18.10, -16.23), (-30.67, -28.80, -25.67), (0.86, 1.10, 1.31)),
        (0, (-64.97, -57.90, -52.26), (0.66, 0.70, 0.78), (0.62, 0.80, 0.95)),
        (0, (-42.42, -39.40, -36.82), (-37.55, -35.70, -32.73), (0.93, 1.20, 1.42)),
        (0, (-102.97, -98.00, -87.20), (-23.90, -20.80, -18.69), (1.32, 1.70, 2.02)),
        (0, (-4.24, -3.80, -3.46), (-54.46, -50.60, -47.85), (0.70, 0.90, 1.07)),
        (0, (-21.78, -19.20, -17.58), (-41.59, -36.70, -32.04), (0.62, 0.80, 0.95)),
        (0, (-69.28, -61.20, -53.74), (-63.25, -56.20, -52.03), (1.32, 1.70, 2.02)),
        (0, (-115.90, -106.40, -99.83), (-25.66, -22.90, -20.74), (1.17, 1.50, 1.78)),
        (0, (-52.76, -49.20, -43.45), (-19.21, -17.20, -15.59), (0.23, 0.30, 0.36)),
        (0, (-176.92, -164.10, -148.67), (-19.34, -17.70, -16.34), (1.01, 1.30, 1.54)),
        (0, (-71.86, -64.70, -60.22), (-4.40, -4.00, -3.74), (0.08, 0.10, 0.12)),
        (0, (-140.44, -129.00, -110.73), (-16.17, -14.20, -12.66), (1.01, 1.30, 1.54)),
        (0, (-135.38, -118.30, -104.84), (-37.25, -34.20, -29.86), (1.17, 1.50, 1.78)),
        (0, (-68.90, -62.80, -55.29), (-98.31, -89.50, -81.50), (1.32, 1.70, 2.02)),
        (1, (11.44, 12.50, 13.95), (6.21, 7.00, 7.81), (1.40, 1.80, 2.14)),
        (1, (19.46, 21.70, 24.65), (-8.35, -7.80, -6.97), (1.24, 1.60, 1.90)),
        (1, (7.53, 8.50, 9.33), (5.39, 5.80, 6.41), (1.17, 1.50, 1.78)),
        (1, (33.94, 39.80, 42.55), (12.50, 13.80, 14.66), (0.93, 1.20, 1.42)),
        (1, (36.91, 43.00, 46.31), (14.66, 16.40, 18.74), (1.01, 1.30, 1.54)),
        (1, (20.17, 21.50, 24.60), (-15.82, -14.40, -12.46), (0.78, 1.00, 1.19)),
        (1, (14.80, 17.40, 18.65), (11.73, 12.60, 13.39), (1.01, 1.30, 1.54)),
        (1, (41.81, 46.70, 51.02), (10.88, 12.60, 14.00), (0.70, 0.90, 1.07)),
        (1, (15.33, 16.30, 17.97), (17.60, 20.40, 21.97), (0.78, 1.00, 1.19)),
        (1, (30.59, 35.30, 37.97), (3.58, 4.20, 4.73), (0.70, 0.90, 1.07)),
        (1, (50.48, 53.50, 58.34), (19.18, 20.60, 21.72), (0.86, 1.10, 1.31)),
        (1, (35.97, 40.60, 44.84), (5.14, 5.80, 6.11), (1.40, 1.80, 2.14)),
        (1, (29.11, 33.00, 36.75), (21.29, 23.60, 26.67), (1.17, 1.50, 1.78)),
        (1, (-3.56, -3.30, -3.10), (3.47, 4.00, 4.55), (2.10, 2.70, 3.20)),
        (1, (22.33, 26.10, 27.95), (9.64, 10.40, 11.44), (1.63, 2.10, 2.49)),
        (1, (29.00, 31.40, 33.97), (14.73, 15.70, 17.84), (1.48, 1.90, 2.25)),
        (1, (33.88, 37.30, 42.31), (29.91, 34.10, 38.84), (1.17, 1.50, 1.78)),
        (1, (46.35, 53.10, 60.71), (6.39, 7.10, 7.84), (1.48, 1.90, 2.25)),
        (1, (31.23, 35.00, 39.14), (17.70, 20.80, 21.84), (1.48, 1.90, 2.25)),
        (1, (51.39, 54.70, 58.04), (13.57, 14.60, 15.54), (1.32, 1.70, 2.02)),
        (1, (30.63, 34.60, 38.21), (23.46, 26.40, 28.68), (1.40, 1.80, 2.14)),
        (1, (18.42, 20.80, 23.19), (10.64, 12.50, 13.78), (1.87, 2.40, 2.85)),
        (1, (41.05, 47.00, 50.24), (14.09, 16.00, 17.73), (1.48, 1.90, 2.25)),
        (1, (60.30, 68.60, 73.65), (11.87, 13.80, 15.28), (1.24, 1.60, 1.90)),
        (1, (34.99, 37.30, 41.43), (28.91, 33.40, 37.54), (2.72, 3.50, 4.15)),
        (1, (55.70, 59.50, 66.62), (6.31, 7.00, 7.95), (1.56, 2.00, 2.37)),
        (1, (32.50, 35.90, 39.01), (23.40, 26.40, 28.46), (1.79, 2.30, 2.73)),
        (1, (17.47, 19.90, 21.93), (25.23, 26.70, 29.34), (1.24, 1.60, 1.90)),
        (1, (34.42, 39.40, 43.84), (27.33, 30.50, 34.15), (1.48, 1.90, 2.25)),
        (1, (43.84, 49.60, 55.90), (21.02, 23.80, 25.19), (1.48, 1.90, 2.25)),
        (1, (16.22, 18.10, 20.58), (12.04, 13.50, 14.20), (3.11, 4.00, 4.75)),
        (1, (43.99, 49.50, 54.86), (22.39, 25.10, 28.69), (2.02, 2.60, 3.09)),
    ]

    X_modal = np.array([[triple[1] for triple in row[1:]] for row in rows], dtype=float)
    X_left = np.array([[triple[1] - triple[0] for triple in row[1:]] for row in rows], dtype=float)
    X_right = np.array([[triple[2] - triple[1] for triple in row[1:]] for row in rows], dtype=float)
    labels = np.array([row[0] for row in rows], dtype=int)

    Y = np.zeros((len(rows), 3), dtype=float)
    Y[labels == 0] = (0.98, 0.08, 0.01)
    Y[labels == 1] = (0.02, 0.01, 0.08)
    return Example1Data(X_modal=X_modal, X_left=X_left, X_right=X_right, Y=Y, feature_names=FEATURES)


def q_matrix(distance: str) -> np.ndarray:
    """Quadratic-form matrix for triangular fuzzy-number distances."""
    dist = distance.upper()
    if dist == "YK":
        lambda0, lambda1, lambda2, lambda3, lambda4 = 3.0, 0.25, 0.25, 0.5, 0.5
    elif dist == "DK":
        lambda0, lambda1, lambda2, lambda3, lambda4 = 1.0, 1.0 / 6.0, 1.0 / 6.0, 0.25, 0.25
    else:
        raise ValueError("distance must be 'YK' or 'DK'")
    return np.array(
        [[lambda0, -lambda4, lambda3], [-lambda4, lambda1, 0.0], [lambda3, 0.0, lambda2]],
        dtype=float,
    )


def squared_distance(residuals: np.ndarray, distance: str) -> np.ndarray:
    q = q_matrix(distance)
    return np.einsum("ij,jk,ik->i", residuals, q, residuals)


def _solve_lsq(A: np.ndarray, b: np.ndarray, lb: np.ndarray, ub: np.ndarray, name: str) -> np.ndarray:
    result = lsq_linear(A, b, bounds=(lb, ub), method="trf", tol=1e-12, max_iter=10000)
    if not result.success:
        raise RuntimeError(f"CQP failed for {name}: {result.message}")
    theta = result.x.copy()
    theta[np.abs(theta) < 1e-10] = 0.0
    return theta


def build_m2_branch(data: Example1Data, distance: str, signs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one fixed-slope-sign branch of model M2.

    Unknown order: ``b, l_b, r_b, a1, a2, a3``.  If ``signs[j] = -1``, then
    the spread covariates are interchanged and multiplied by ``-1`` so that a
    negative slope contributes nonnegative spread width.
    """
    X, Xl, Xr, Y = data.X_modal, data.X_left, data.X_right, data.Y
    n, p = X.shape
    q = q_matrix(distance)
    rchol = cholesky(q, lower=False)
    left_cov = np.where(signs == 1, Xl, -Xr)
    right_cov = np.where(signs == 1, Xr, -Xl)

    m = 3 + p
    A = np.zeros((3 * n, m), dtype=float)
    b = np.zeros(3 * n, dtype=float)
    for i in range(n):
        design = np.zeros((3, m), dtype=float)
        design[0, 0] = 1.0
        design[1, 1] = 1.0
        design[2, 2] = 1.0
        design[0, 3:] = X[i]
        design[1, 3:] = left_cov[i]
        design[2, 3:] = right_cov[i]
        sl = slice(3 * i, 3 * i + 3)
        A[sl, :] = rchol @ design
        b[sl] = rchol @ Y[i]

    lb = np.full(m, -np.inf)
    ub = np.full(m, np.inf)
    lb[1] = 0.0
    lb[2] = 0.0
    for j, sj in enumerate(signs):
        if sj == 1:
            lb[3 + j] = 0.0
        else:
            ub[3 + j] = 0.0
    return A, b, lb, ub, left_cov, right_cov


def solve_m2_cqp(data: Example1Data, distance: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve model M2 by enumerating the fixed slope-sign branches."""
    best: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for signs_tuple in itertools.product((-1, 1), repeat=N_FEATURES):
        signs = np.array(signs_tuple, dtype=int)
        A, b, lb, ub, left_cov, right_cov = build_m2_branch(data, distance, signs)
        theta = _solve_lsq(A, b, lb, ub, name=f"M2 {distance} signs={signs_tuple}")
        pred = predict_m2(data, theta, left_cov, right_cov)
        objective = float(np.sum(squared_distance(pred - data.Y, distance)))
        if best is None or objective < best[0]:
            best = (objective, theta, signs, left_cov, right_cov)
    assert best is not None
    _, theta, signs, left_cov, right_cov = best
    return theta, signs, predict_m2(data, theta, left_cov, right_cov)


def predict_m2(data: Example1Data, theta: np.ndarray, left_cov: np.ndarray, right_cov: np.ndarray) -> np.ndarray:
    b, lb, rb = theta[:3]
    a = theta[3:]
    return np.column_stack([
        b + data.X_modal @ a,
        lb + left_cov @ a,
        rb + right_cov @ a,
    ])


def build_m3_cqp(data: Example1Data, distance: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the signed-input CQP for model M3.

    Unknown order: ``b, l_b, r_b, a1, l_a1, r_a1, ..., a3, l_a3, r_a3``.
    For signed crisp inputs, the left-spread design uses
    ``x^+ l_a + x^- r_a`` and the right-spread design uses
    ``x^+ r_a + x^- l_a``.
    """
    X, Y = data.X_modal, data.Y
    n, p = X.shape
    q = q_matrix(distance)
    rchol = cholesky(q, lower=False)
    m = 3 * (p + 1)
    A = np.zeros((3 * n, m), dtype=float)
    b = np.zeros(3 * n, dtype=float)
    for i in range(n):
        design = np.zeros((3, m), dtype=float)
        design[0, 0] = 1.0
        design[1, 1] = 1.0
        design[2, 2] = 1.0
        for j in range(p):
            base = 3 + 3 * j
            x = X[i, j]
            xp = max(x, 0.0)
            xn = max(-x, 0.0)
            design[0, base] = x
            design[1, base + 1] = xp
            design[1, base + 2] = xn
            design[2, base + 1] = xn
            design[2, base + 2] = xp
        sl = slice(3 * i, 3 * i + 3)
        A[sl, :] = rchol @ design
        b[sl] = rchol @ Y[i]

    lb = np.full(m, -np.inf)
    ub = np.full(m, np.inf)
    lb[1] = 0.0
    lb[2] = 0.0
    for j in range(p):
        base = 3 + 3 * j
        lb[base + 1] = 0.0
        lb[base + 2] = 0.0
    return A, b, lb, ub


def solve_m3_cqp(data: Example1Data, distance: str) -> tuple[np.ndarray, np.ndarray]:
    A, b, lb, ub = build_m3_cqp(data, distance)
    theta = _solve_lsq(A, b, lb, ub, name=f"M3 {distance}")
    return theta, predict_m3(data, theta)


def predict_m3(data: Example1Data, theta: np.ndarray) -> np.ndarray:
    X = data.X_modal
    p = X.shape[1]
    b, lb, rb = theta[:3]
    a = np.array([theta[3 + 3 * j] for j in range(p)])
    l = np.array([theta[3 + 3 * j + 1] for j in range(p)])
    r = np.array([theta[3 + 3 * j + 2] for j in range(p)])
    Xp = np.maximum(X, 0.0)
    Xn = np.maximum(-X, 0.0)
    return np.column_stack([
        b + X @ a,
        lb + Xp @ l + Xn @ r,
        rb + Xn @ l + Xp @ r,
    ])


def metrics(data: Example1Data, pred: np.ndarray) -> Dict[str, float]:
    n = len(data.Y)
    p = N_FEATURES
    y = data.Y
    ybar = y.mean(axis=0)
    out: Dict[str, float] = {}
    for dist in ("YK", "DK"):
        sse = float(np.sum(squared_distance(pred - y, dist)))
        sst = float(np.sum(squared_distance(y - ybar, dist)))
        out[f"S_D{dist}2"] = sse / (n - p - 1)
        out[f"R_D{dist}2"] = 1.0 - sse / sst
    return out


def coefficient_rows_m2(theta: np.ndarray, signs: np.ndarray, fit_distance: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = [
        {"model": "M2", "fit_distance": fit_distance, "term": "intercept", "modal": theta[0], "left_spread": theta[1], "right_spread": theta[2], "slope_sign": ""}
    ]
    for j, name in enumerate(FEATURES):
        rows.append({"model": "M2", "fit_distance": fit_distance, "term": name, "modal": theta[3 + j], "left_spread": np.nan, "right_spread": np.nan, "slope_sign": int(signs[j])})
    return rows


def coefficient_rows_m3(theta: np.ndarray, fit_distance: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = [
        {"model": "M3", "fit_distance": fit_distance, "term": "intercept", "modal": theta[0], "left_spread": theta[1], "right_spread": theta[2]}
    ]
    for j, name in enumerate(FEATURES):
        base = 3 + 3 * j
        rows.append({"model": "M3", "fit_distance": fit_distance, "term": name, "modal": theta[base], "left_spread": theta[base + 1], "right_spread": theta[base + 2]})
    return rows


def make_paper_table(m2_metrics: pd.DataFrame, m3_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the Table 2 values in the manuscript."""
    m2_dk = m2_metrics[m2_metrics["fit_distance"] == "DK"].iloc[0]
    m2_yk = m2_metrics[m2_metrics["fit_distance"] == "YK"].iloc[0]
    m3_combined = {
        "S_DYK2": min(m3_metrics["S_DYK2"]),
        "S_DDK2": min(m3_metrics["S_DDK2"]),
        "R_DYK2": max(m3_metrics["R_DYK2"]),
        "R_DDK2": max(m3_metrics["R_DDK2"]),
    }
    rows = [
        {"Model": "The model M2 fitted with D_DK", **{k: round(float(m2_dk[k]), 4) for k in ["S_DYK2", "S_DDK2", "R_DYK2", "R_DDK2"]}},
        {"Model": "The model M2 fitted with D_YK", **{k: round(float(m2_yk[k]), 4) for k in ["S_DYK2", "S_DDK2", "R_DYK2", "R_DDK2"]}},
        {"Model": "The model M3 fitted with D_DK or D_YK", **{k: round(float(v), 4) for k, v in m3_combined.items()}},
        {"Model": "The classical regression model", "S_DYK2": 0.1935, "S_DDK2": 0.0635, "R_DYK2": 0.7233, "R_DDK2": 0.7209},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    data = load_example1_data()

    m2_coeff_records: list[dict[str, float | str]] = []
    m2_metric_records: list[dict[str, float | str]] = []
    m3_coeff_records: list[dict[str, float | str]] = []
    m3_metric_records: list[dict[str, float | str]] = []

    for fit_distance in ("DK", "YK"):
        theta2, signs2, pred2 = solve_m2_cqp(data, fit_distance)
        m2_coeff_records.extend(coefficient_rows_m2(theta2, signs2, fit_distance))
        row2: dict[str, float | str] = {"fit_distance": fit_distance}
        row2.update(metrics(data, pred2))
        m2_metric_records.append(row2)

        theta3, pred3 = solve_m3_cqp(data, fit_distance)
        m3_coeff_records.extend(coefficient_rows_m3(theta3, fit_distance))
        row3: dict[str, float | str] = {"fit_distance": fit_distance}
        row3.update(metrics(data, pred3))
        m3_metric_records.append(row3)

    m2_coeff_df = pd.DataFrame(m2_coeff_records)
    m2_metric_df = pd.DataFrame(m2_metric_records)
    m3_coeff_df = pd.DataFrame(m3_coeff_records)
    m3_metric_df = pd.DataFrame(m3_metric_records)
    table_df = make_paper_table(m2_metric_df, m3_metric_df)

    m2_coeff_df.to_csv(OUTDIR / "example1_constrained_m2_coefficients.csv", index=False)
    m2_metric_df.to_csv(OUTDIR / "example1_constrained_m2_metrics.csv", index=False)
    m3_coeff_df.to_csv(OUTDIR / "example1_constrained_m3_coefficients.csv", index=False)
    m3_metric_df.to_csv(OUTDIR / "example1_constrained_m3_metrics.csv", index=False)
    table_df.to_csv(OUTDIR / "example1_table2_revised_constrained.csv", index=False)

    print("Example 1 constrained model M2 metrics")
    print(m2_metric_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nExample 1 constrained model M3 metrics")
    print(m3_metric_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nManuscript Table 2 values")
    print(table_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
