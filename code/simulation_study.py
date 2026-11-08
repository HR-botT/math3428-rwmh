"""Main study: MSE and ESS vs proposal scale across dimensions.

Design:
  dims        : 1, 5, 20
  scales      : 12 log-spaced values from 0.15*sigma* to 8*sigma*
                where sigma* = 2.38/sqrt(d)
  reps        : 100 independent chains per (d, scale)
  iterations  : 1500 after 500 burn-in
Recorded per run: acceptance rate, per-dim mean estimate, per-dim ESS.
"""
from __future__ import annotations
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "logs"); os.makedirs(LOG, exist_ok=True)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from algorithm import RandomWalkMH
from targets import gaussian_log_target, identity_cov
from diagnostics import ess

DIMS = [1, 5, 20]
N_SCALES = 12
REPS = 100
N_ITER = 1500
BURN_IN = 500
BASE_SEED = 7_000_000


def main():
    t0 = time.time()
    raw_rows = []
    for d in DIMS:
        sigma_star = 2.38 / np.sqrt(d)
        scales = np.geomspace(0.15 * sigma_star, 8.0 * sigma_star, N_SCALES)
        log_target = gaussian_log_target(np.zeros(d), identity_cov(d))
        for k, sigma in enumerate(scales):
            for rep in range(REPS):
                rng = np.random.default_rng(BASE_SEED + 10_000 * d + 100 * k + rep)
                mh = RandomWalkMH(log_target, d, sigma, rng)
                chain, acc = mh.sample(np.zeros(d), n_iter=N_ITER, burn_in=BURN_IN)
                mean_est = chain.mean(axis=0)
                mse_rep = float(np.mean(mean_est ** 2))   # truth is zero
                ess_rep = float(np.mean([ess(chain[:, j]) for j in range(d)]))
                raw_rows.append({
                    "d": d, "scale": sigma, "rep": rep,
                    "accept_rate": acc, "mse_rep": mse_rep,
                    "ess_mean_rep": ess_rep,
                })
            if (k + 1) % 4 == 0:
                print(f"  d={d:>2}, scale {k+1}/{N_SCALES} done "
                      f"({time.time()-t0:.1f}s)")

    raw = pd.DataFrame(raw_rows)
    raw_path = os.path.join(OUT, "results_simulation.csv")
    raw.to_csv(raw_path, index=False)

    # ---- Summary ---------------------------------------------------------
    g = raw.groupby(["d", "scale"], as_index=False)
    summary = g.agg(
        acc_rate=("accept_rate", "mean"),
        mse=("mse_rep", "mean"),
        mse_se=("mse_rep", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        ess_mean=("ess_mean_rep", "mean"),
        ess_mean_se=("ess_mean_rep", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
    )
    summary["sigma_star"] = 2.38 / np.sqrt(summary["d"])
    summary["rel_scale"] = summary["scale"] / summary["sigma_star"]
    summ_path = os.path.join(OUT, "results_simulation_summary.csv")
    summary.to_csv(summ_path, index=False)

    print(f"Saved: {raw_path}\nSaved: {summ_path}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()