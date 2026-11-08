"""Baseline comparison: MH at optimal scale vs i.i.d. sampling.

For each dimension, compare at matched computational budget:
  - i.i.d. draws from the target (gold-standard baseline, but cheating)
  - RWMH at sigma* = 2.38/sqrt(d)
Report MSE of the mean, ESS, and wall-clock runtime.
"""
from __future__ import annotations
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from algorithm import RandomWalkMH
from targets import gaussian_log_target, identity_cov
from diagnostics import ess

DIMS = [1, 5, 20]
REPS = 200
N_ITER = 1500
BURN_IN = 500
BASE_SEED = 8_000_000


def run_mh(d, sigma, rng):
    log_t = gaussian_log_target(np.zeros(d), identity_cov(d))
    t0 = time.perf_counter()
    mh = RandomWalkMH(log_t, d, sigma, rng)
    chain, acc = mh.sample(np.zeros(d), n_iter=N_ITER, burn_in=BURN_IN)
    rt = time.perf_counter() - t0
    mean_est = chain.mean(axis=0)
    mse = float(np.mean(mean_est ** 2))
    ess_v = float(np.mean([ess(chain[:, j]) for j in range(d)]))
    return acc, mse, ess_v, rt


def run_iid(d, rng):
    t0 = time.perf_counter()
    draws = rng.standard_normal((N_ITER, d))
    rt = time.perf_counter() - t0
    mean_est = draws.mean(axis=0)
    mse = float(np.mean(mean_est ** 2))
    # i.i.d. -> ESS = n by definition
    return 1.0, mse, float(N_ITER), rt


def main():
    rows = []
    for d in DIMS:
        sigma_star = 2.38 / np.sqrt(d)
        for rep in range(REPS):
            seed = BASE_SEED + 1_000 * d + rep
            rng = np.random.default_rng(seed)
            acc, mse, ess_v, rt = run_mh(d, sigma_star, rng)
            rows.append({"d": d, "method": "RWMH(optimal)", "rep": rep,
                         "accept_rate": acc, "mse": mse,
                         "ess_mean": ess_v, "runtime_s": rt})

            rng = np.random.default_rng(seed + 1)
            acc, mse, ess_v, rt = run_iid(d, rng)
            rows.append({"d": d, "method": "iid", "rep": rep,
                         "accept_rate": acc, "mse": mse,
                         "ess_mean": ess_v, "runtime_s": rt})

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(OUT, "results_baseline.csv"), index=False)

    summary = raw.groupby(["d", "method"], as_index=False).agg(
        mse=("mse", "mean"),
        mse_se=("mse", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        ess_mean=("ess_mean", "mean"),
        runtime_s=("runtime_s", "mean"),
        accept_rate=("accept_rate", "mean"),
    )
    iid_mse = summary[summary["method"] == "iid"].set_index("d")["mse"]
    summary["mse_ratio_vs_iid"] = summary["mse"] / summary["d"].map(iid_mse)
    summary.to_csv(os.path.join(OUT, "results_baseline_summary.csv"), index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved: {os.path.join(OUT, 'results_baseline.csv')}")
    print(f"Saved: {os.path.join(OUT, 'results_baseline_summary.csv')}")


if __name__ == "__main__":
    main()