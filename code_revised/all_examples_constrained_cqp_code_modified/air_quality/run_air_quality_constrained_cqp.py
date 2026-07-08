#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Air-quality constrained CQP calculation for the revised manuscript.

The script reads the hourly CSV file, aggregates it into city-day LR-type fuzzy
observations, solves the nonnegative-spread constrained quadratic programs for
models M2 and M3, and writes the coefficient and model-comparison tables used in
Table 6 of the revised manuscript.

Default input
-------------
data/air_quality_2024.csv

Outputs
-------
outputs/daily_fuzzy_observations.csv
outputs/air_quality_constrained_coefficients.csv
outputs/air_quality_constrained_metrics.csv
outputs/air_quality_table6_revised_constrained.csv

Dependencies
------------
numpy, pandas, scipy
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.optimize import lsq_linear

FEATURES = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10"]
TARGET = "AQI"
N_FEATURES = len(FEATURES)


def default_csv_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "air_quality_2024.csv"


def q_matrix(shape: str, distance: str) -> np.ndarray:
    """Return the 3x3 quadratic matrix for D_YK or D_DK.

    The component order is (modal residual, left-spread residual,
    right-spread residual).  The constants are those used in the revised
    manuscript for triangular and normal LR-type fuzzy numbers.
    """
    shape = shape.upper()
    distance = distance.upper()
    if shape == "T":
        L0 = R0 = 0.5
        L1 = R1 = 0.25
        L2 = R2 = 1.0 / 6.0
    elif shape == "N":
        L0 = R0 = math.sqrt(math.pi) / 2.0
        L1 = R1 = math.sqrt(math.pi) / 4.0
        L2 = R2 = 0.5
    else:
        raise ValueError("shape must be 'T' or 'N'")

    if distance == "YK":
        lambda0, lambda1, lambda2, lambda3, lambda4 = 3.0, L0 * L0, R0 * R0, R0, L0
    elif distance == "DK":
        lambda0, lambda1, lambda2, lambda3, lambda4 = 1.0, L2, R2, R1, L1
    else:
        raise ValueError("distance must be 'YK' or 'DK'")

    return np.array(
        [[lambda0, -lambda4, lambda3], [-lambda4, lambda1, 0.0], [lambda3, 0.0, lambda2]],
        dtype=float,
    )


def squared_distance(residuals: np.ndarray, shape: str, distance: str) -> np.ndarray:
    q = q_matrix(shape, distance)
    return np.einsum("ij,jk,ik->i", residuals, q, residuals)


def load_hourly_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    missing = {"Date", "City", *FEATURES, TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    df = df.dropna(subset=["Date", "City"] + FEATURES + [TARGET]).copy()
    df["day"] = df["Date"].dt.date
    return df


def aggregate_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    cols = FEATURES + [TARGET]
    daily = df.groupby(["City", "day"])[cols].agg(["mean", "min", "max", "std"]).reset_index()

    def arr(col: str, stat: str) -> np.ndarray:
        return daily[(col, stat)].to_numpy(dtype=float)

    X_mean = np.column_stack([arr(c, "mean") for c in FEATURES])
    X_min = np.column_stack([arr(c, "min") for c in FEATURES])
    X_max = np.column_stack([arr(c, "max") for c in FEATURES])
    X_std = np.column_stack([arr(c, "std") for c in FEATURES])
    Y_mean = arr(TARGET, "mean")
    Y_min = arr(TARGET, "min")
    Y_max = arr(TARGET, "max")
    Y_std = arr(TARGET, "std")

    arrays = {
        "X_mean": X_mean,
        "X_l": X_mean - X_min,
        "X_r": X_max - X_mean,
        "X_s": X_std,
        "Y_mean": Y_mean,
        "Y_l": Y_mean - Y_min,
        "Y_r": Y_max - Y_mean,
        "Y_s": Y_std,
    }

    flat = daily[["City", "day"]].copy()
    for c in FEATURES + [TARGET]:
        flat[f"{c}_mean"] = arr(c, "mean")
        flat[f"{c}_min"] = arr(c, "min")
        flat[f"{c}_max"] = arr(c, "max")
        flat[f"{c}_std"] = arr(c, "std")
    return flat, arrays


def observed(arrays: Dict[str, np.ndarray], shape: str) -> np.ndarray:
    if shape.upper() == "T":
        return np.column_stack([arrays["Y_mean"], arrays["Y_l"], arrays["Y_r"]])
    return np.column_stack([arrays["Y_mean"], arrays["Y_s"], arrays["Y_s"]])


def build_m2(arrays: Dict[str, np.ndarray], shape: str, fit_distance: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build CQP for model M2.

    Unknown order:
        T: b, l_b, r_b, a1,...,a6
        N: b, sigma_b, a1,...,a6
    Only fitted fuzzy-coefficient spreads are constrained.  In model M2, the
    slopes are real scalars and have no spread components.
    """
    q = q_matrix(shape, fit_distance)
    r = cholesky(q, lower=False)
    n = len(arrays["Y_mean"])
    p = len(FEATURES)

    if shape.upper() == "T":
        m = 3 + p
        y = observed(arrays, "T")
        A = np.zeros((3 * n, m), dtype=float)
        b = np.zeros(3 * n, dtype=float)
        for i in range(n):
            design = np.zeros((3, m), dtype=float)
            design[0, 0] = 1.0
            design[1, 1] = 1.0
            design[2, 2] = 1.0
            design[0, 3:] = arrays["X_mean"][i]
            design[1, 3:] = arrays["X_l"][i]
            design[2, 3:] = arrays["X_r"][i]
            sl = slice(3 * i, 3 * i + 3)
            A[sl, :] = r @ design
            b[sl] = r @ y[i]
        lb = np.full(m, -np.inf)
        ub = np.full(m, np.inf)
        lb[1] = 0.0
        lb[2] = 0.0
    else:
        m = 2 + p
        y = observed(arrays, "N")
        A = np.zeros((3 * n, m), dtype=float)
        b = np.zeros(3 * n, dtype=float)
        for i in range(n):
            design = np.zeros((3, m), dtype=float)
            design[0, 0] = 1.0
            design[1, 1] = 1.0
            design[2, 1] = 1.0
            design[0, 2:] = arrays["X_mean"][i]
            design[1, 2:] = arrays["X_s"][i]
            design[2, 2:] = arrays["X_s"][i]
            sl = slice(3 * i, 3 * i + 3)
            A[sl, :] = r @ design
            b[sl] = r @ y[i]
        lb = np.full(m, -np.inf)
        ub = np.full(m, np.inf)
        lb[1] = 0.0
    return A, b, lb, ub


def build_m3(arrays: Dict[str, np.ndarray], shape: str, fit_distance: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build CQP for model M3.

    Unknown order:
        T: b,l_b,r_b,a1,l_a1,r_a1,...,a6,l_a6,r_a6
        N: b,sigma_b,a1,sigma_a1,...,a6,sigma_a6
    """
    q = q_matrix(shape, fit_distance)
    r = cholesky(q, lower=False)
    X = arrays["X_mean"]
    n, p = X.shape

    if shape.upper() == "T":
        m = 3 * (p + 1)
        y = observed(arrays, "T")
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
            A[sl, :] = r @ design
            b[sl] = r @ y[i]
        lb = np.full(m, -np.inf)
        ub = np.full(m, np.inf)
        lb[1] = 0.0
        lb[2] = 0.0
        for j in range(p):
            base = 3 + 3 * j
            lb[base + 1] = 0.0
            lb[base + 2] = 0.0
    else:
        m = 2 * (p + 1)
        y = observed(arrays, "N")
        A = np.zeros((3 * n, m), dtype=float)
        b = np.zeros(3 * n, dtype=float)
        for i in range(n):
            design = np.zeros((3, m), dtype=float)
            design[0, 0] = 1.0
            design[1, 1] = 1.0
            design[2, 1] = 1.0
            for j in range(p):
                base = 2 + 2 * j
                x = X[i, j]
                design[0, base] = x
                design[1, base + 1] = abs(x)
                design[2, base + 1] = abs(x)
            sl = slice(3 * i, 3 * i + 3)
            A[sl, :] = r @ design
            b[sl] = r @ y[i]
        lb = np.full(m, -np.inf)
        ub = np.full(m, np.inf)
        lb[1] = 0.0
        for j in range(p):
            lb[2 + 2 * j + 1] = 0.0
    return A, b, lb, ub


def solve_cqp(A: np.ndarray, b: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    result = lsq_linear(A, b, bounds=(lb, ub), method="trf", tol=1e-12, max_iter=10000)
    if not result.success:
        raise RuntimeError(result.message)
    theta = result.x.copy()
    theta[np.abs(theta) < 1e-8] = 0.0
    return theta


def predict(arrays: Dict[str, np.ndarray], model: str, shape: str, theta: np.ndarray) -> np.ndarray:
    X = arrays["X_mean"]
    if model == "M2" and shape == "T":
        b, lb, rb = theta[:3]
        a = theta[3:]
        return np.column_stack([b + X @ a, lb + arrays["X_l"] @ a, rb + arrays["X_r"] @ a])
    if model == "M2" and shape == "N":
        b, sb = theta[:2]
        a = theta[2:]
        sigma = sb + arrays["X_s"] @ a
        return np.column_stack([b + X @ a, sigma, sigma])
    if model == "M3" and shape == "T":
        b, lb, rb = theta[:3]
        a = np.array([theta[3 + 3 * j] for j in range(N_FEATURES)])
        l = np.array([theta[3 + 3 * j + 1] for j in range(N_FEATURES)])
        r = np.array([theta[3 + 3 * j + 2] for j in range(N_FEATURES)])
        Xp = np.maximum(X, 0.0)
        Xn = np.maximum(-X, 0.0)
        return np.column_stack([b + X @ a, lb + Xp @ l + Xn @ r, rb + Xn @ l + Xp @ r])
    if model == "M3" and shape == "N":
        b, sb = theta[:2]
        a = np.array([theta[2 + 2 * j] for j in range(N_FEATURES)])
        s = np.array([theta[2 + 2 * j + 1] for j in range(N_FEATURES)])
        sigma = sb + np.abs(X) @ s
        return np.column_stack([b + X @ a, sigma, sigma])
    raise ValueError((model, shape))


def metrics(arrays: Dict[str, np.ndarray], pred: np.ndarray, shape: str) -> Dict[str, float]:
    y = observed(arrays, shape)
    ybar = y.mean(axis=0)
    n = len(y)
    out: Dict[str, float] = {}
    for dist in ("YK", "DK"):
        sse = float(np.sum(squared_distance(pred - y, shape, dist)))
        sst = float(np.sum(squared_distance(y - ybar, shape, dist)))
        out[f"S_D{dist}2"] = sse / (n - N_FEATURES - 1)
        out[f"R_D{dist}2"] = 1.0 - sse / sst
    return out


def coefficient_rows(model: str, shape: str, fit_distance: str, theta: np.ndarray) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if model == "M2" and shape == "T":
        rows.append({"model": model, "shape": shape, "fit_distance": fit_distance, "term": "intercept", "modal": theta[0], "left_spread": theta[1], "right_spread": theta[2]})
        for j, f in enumerate(FEATURES):
            rows.append({"model": model, "shape": shape, "fit_distance": fit_distance, "term": f, "modal": theta[3 + j], "left_spread": np.nan, "right_spread": np.nan})
    elif model == "M2" and shape == "N":
        rows.append({"model": model, "shape": shape, "fit_distance": fit_distance, "term": "intercept", "modal": theta[0], "sigma": theta[1]})
        for j, f in enumerate(FEATURES):
            rows.append({"model": model, "shape": shape, "fit_distance": fit_distance, "term": f, "modal": theta[2 + j], "sigma": np.nan})
    elif model == "M3" and shape == "T":
        rows.append({"model": model, "shape": shape, "fit_distance": fit_distance, "term": "intercept", "modal": theta[0], "left_spread": theta[1], "right_spread": theta[2]})
        for j, f in enumerate(FEATURES):
            base = 3 + 3 * j
            rows.append({"model": model, "shape": shape, "fit_distance": fit_distance, "term": f, "modal": theta[base], "left_spread": theta[base + 1], "right_spread": theta[base + 2]})
    elif model == "M3" and shape == "N":
        rows.append({"model": model, "shape": shape, "fit_distance": fit_distance, "term": "intercept", "modal": theta[0], "sigma": theta[1]})
        for j, f in enumerate(FEATURES):
            base = 2 + 2 * j
            rows.append({"model": model, "shape": shape, "fit_distance": fit_distance, "term": f, "modal": theta[base], "sigma": theta[base + 1]})
    return rows


def make_table6(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Return the manuscript Table 6 values.

    The proposed-model rows are computed by this script.  The external
    benchmark rows are the values reported in the manuscript for the same
    aggregated triangular fuzzy response.
    """
    table_rows = []
    order = [
        ("M2", "T", "DK", "Model $\\mathbf{M2}$ fitted with $D_{DK}$ and TFNs"),
        ("M2", "T", "YK", "Model $\\mathbf{M2}$ fitted with $D_{YK}$ and TFNs"),
        ("M2", "N", "DK", "Model $\\mathbf{M2}$ fitted with $D_{DK}$ and NFNs"),
        ("M2", "N", "YK", "Model $\\mathbf{M2}$ fitted with $D_{YK}$ and NFNs"),
        ("M3", "T", "DK", "Model $\\mathbf{M3}$ fitted with $D_{DK}$ and TFNs"),
        ("M3", "T", "YK", "Model $\\mathbf{M3}$ fitted with $D_{YK}$ and TFNs"),
        ("M3", "N", "DK", "Model $\\mathbf{M3}$ fitted with NFNs"),
    ]
    for model, shape, fit, label in order:
        row = metric_df[(metric_df["model"] == model) & (metric_df["shape"] == shape) & (metric_df["fit_distance"] == fit)].iloc[0]
        table_rows.append({
            "Model": label,
            "S_DYK2": round(float(row["S_DYK2"]), 2),
            "S_DDK2": round(float(row["S_DDK2"]), 2),
            "R_DYK2": round(float(row["R_DYK2"]), 4),
            "R_DDK2": round(float(row["R_DDK2"]), 4),
        })
    table_rows.extend(
        [
            {"Model": "Classical multiple least-squares regression", "S_DYK2": 189.00, "S_DDK2": 63.00, "R_DYK2": 0.9051, "R_DDK2": 0.9053},
            {"Model": "The two-stage fuzzy linear regression", "S_DYK2": 276.93, "S_DDK2": 114.99, "R_DYK2": 0.8604, "R_DDK2": 0.8271},
            {"Model": "The weighted least-squares fuzzy regression", "S_DYK2": 180.29, "S_DDK2": 62.82, "R_DYK2": 0.9091, "R_DDK2": 0.9055},
            {"Model": "The least-median-squares robust fuzzy regression", "S_DYK2": 189.82, "S_DDK2": 66.03, "R_DYK2": 0.9043, "R_DDK2": 0.9007},
            {"Model": "The alpha-cut interval least-squares fuzzy regression", "S_DYK2": 290.30, "S_DDK2": 119.32, "R_DYK2": 0.8537, "R_DDK2": 0.8206},
            {"Model": "The robust fuzzy-random-variable regression", "S_DYK2": 325.51, "S_DDK2": 138.12, "R_DYK2": 0.8360, "R_DDK2": 0.7923},
        ]
    )
    return pd.DataFrame(table_rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute constrained CQP rows for the air-quality example.")
    parser.add_argument("--csv", type=Path, default=default_csv_path(), help="Hourly air-quality CSV path.")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "outputs", help="Output directory.")
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    hourly = load_hourly_csv(args.csv)
    daily, arrays = aggregate_daily(hourly)
    daily.to_csv(args.outdir / "daily_fuzzy_observations.csv", index=False)

    coeff_records = []
    metric_records = []
    for model in ("M2", "M3"):
        for shape in ("T", "N"):
            for fit_distance in ("DK", "YK"):
                if model == "M2":
                    A, b, lb, ub = build_m2(arrays, shape, fit_distance)
                else:
                    A, b, lb, ub = build_m3(arrays, shape, fit_distance)
                theta = solve_cqp(A, b, lb, ub)
                pred_values = predict(arrays, model, shape, theta)
                row = {"model": model, "shape": shape, "fit_distance": fit_distance}
                row.update(metrics(arrays, pred_values, shape))
                metric_records.append(row)
                coeff_records.extend(coefficient_rows(model, shape, fit_distance, theta))

    metric_df = pd.DataFrame(metric_records)
    coeff_df = pd.DataFrame(coeff_records)
    table6_df = make_table6(metric_df)

    coeff_df.to_csv(args.outdir / "air_quality_constrained_coefficients.csv", index=False)
    metric_df.to_csv(args.outdir / "air_quality_constrained_metrics.csv", index=False)
    table6_df.to_csv(args.outdir / "air_quality_table6_revised_constrained.csv", index=False)

    print(f"Hourly rows: {len(hourly):,}")
    print(f"City-day observations: {len(daily):,}")
    print("\nRecomputed constrained Table 6 rows:")
    print(table6_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
