"""Computational budget condition: MSE vs number of iterations.

At optimal scale (d=20), vary N in {250, 500, 1000, 2000, 4000, 8000}
and check the expected 1/N decay of the MSE of the posterior mean.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from algorithm import RandomWalkMH
from targets import gaussian_log_target, identity_cov
from diagnostics import ess

D = 20
N_VALUES = [250, 500, 1000, 2000, 4000, 8000]
BURN_IN = 500
REPS = 200
BASE_SEED = 9_000_000


def main():
    sigma_star = 2.38 / np.sqrt(D)
    log_t = gaussian_log_target(np.zeros(D), identity_cov(D))
    rows = []
    for N in N_VALUES:
        for rep in range(REPS):
            rng = np.random.default_rng(BASE_SEED + 100 * N + rep)
            mh = RandomWalkMH(log_t, D, sigma_star, rng)
            chain, acc = mh.sample(np.zeros(D), n_iter=N, burn_in=BURN_IN)
            mean_est = chain.mean(axis=0)
            mse = float(np.mean(mean_est ** 2))
            ess_v = float(np.mean([ess(chain[:, j]) for j in range(D)]))
            rows.append({"d": D, "n_iter": N, "rep": rep,
                         "mse": mse, "ess_mean": ess_v, "accept_rate": acc})
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(OUT, "results_budget.csv"), index=False)
    summary = raw.groupby("n_iter", as_index=False).agg(
        mse=("mse", "mean"),
        mse_se=("mse", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        ess_mean=("ess_mean", "mean"),
        accept_rate=("accept_rate", "mean"),
    )
    summary["mse_x_n"] = summary["mse"] * summary["n_iter"]
    summary.to_csv(os.path.join(OUT, "results_budget_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()