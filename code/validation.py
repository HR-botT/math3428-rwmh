"""Validate the RWMH implementation against known results.

Target: N(0, I_5). Known truth: mean 0, covariance I_5, marginals N(0,1).
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

import numpy as np
from scipy import stats
from algorithm import RandomWalkMH
from targets import gaussian_log_target, identity_cov
from diagnostics import ess


def main(seed=20240101, d=5, n_iter=200_000, burn_in=2_000):
    rng = np.random.default_rng(seed)
    log_target = gaussian_log_target(np.zeros(d), identity_cov(d))
    sigma_star = 2.38 / np.sqrt(d)  # classical optimal scale for product target

    mh = RandomWalkMH(log_target, d, sigma_star, rng)
    chain, acc = mh.sample(np.zeros(d), n_iter=n_iter, burn_in=burn_in)

    res = {"dim": d, "scale": sigma_star, "acceptance_rate": acc,
           "n_draws": int(chain.shape[0])}

    # ---- Mean and covariance ---------------------------------------------
    sample_mean = chain.mean(axis=0)
    ess_per_dim = np.array([ess(chain[:, j]) for j in range(d)])
    mcse = np.array([
        chain[:, j].std(ddof=1) / np.sqrt(ess_per_dim[j]) for j in range(d)
    ])
    res["sample_mean"] = sample_mean.tolist()
    res["mcse_of_mean"] = mcse.tolist()
    res["mean_within_4mcse_of_truth"] = bool(np.all(np.abs(sample_mean) < 4 * mcse))

    emp_cov = np.cov(chain, rowvar=False)
    res["sample_var_diag"] = np.diag(emp_cov).tolist()
    off = emp_cov - np.diag(np.diag(emp_cov))
    res["max_abs_offdiag"] = float(np.max(np.abs(off)))

    # ---- Marginal KS tests vs N(0,1) -------------------------------------
    ks = []
    for j in range(d):
        stat, p = stats.kstest(chain[:, j], "norm")
        ks.append({"dim": j, "ks_stat": float(stat), "p_value": float(p)})
    res["ks_tests"] = ks

        # Thinned KS test (approximately independent draws).
    # Raw-draw KS is over-powered when draws are autocorrelated; thinning to
    # approximately the ESS gives a valid test on near-independent draws.
    ks_thin = []
    for j in range(d):
        tf = max(1, int(np.ceil(chain.shape[0] / ess_per_dim[j])))
        thinned_j = chain[::tf, j]
        stat, p = stats.kstest(thinned_j, "norm")
        ks_thin.append({
            "dim": j, "thin": tf, "n_thinned": int(len(thinned_j)),
            "ks_stat": float(stat), "p_value": float(p),
        })
    res["ks_tests_thinned"] = ks_thin

    # ---- Compare with iid sampling ---------------------------------------
    iid = rng.standard_normal((chain.shape[0], d))
    res["iid_sample_mean"] = iid.mean(axis=0).tolist()

    # ---- Save ------------------------------------------------------------
    path = os.path.join(OUT, "validation.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)

    print("Validation summary (target N(0, I_5), scale = 2.38/sqrt(5))")
    print(f"  acceptance rate        : {acc:.3f}")
    print(f"  draws                  : {chain.shape[0]}")
    print(f"  sample means           : {np.round(sample_mean, 4)}")
    print(f"  MCSE of means          : {np.round(mcse, 4)}")
    print(f"  means within 4*MCSE    : {res['mean_within_4mcse_of_truth']}")
    print(f"  sample variances (diag): {np.round(res['sample_var_diag'], 4)}")
    print(f"  max |off-diagonal cov| : {res['max_abs_offdiag']:.4f}")
    print("  marginal KS tests vs N(0,1):")
    for row in ks:
        print(f"    dim {row['dim']}: stat={row['ks_stat']:.4f}, p={row['p_value']:.3f}")
    print("  thinned marginal KS tests vs N(0,1):")
    for row in ks_thin:
        print(f"    dim {row['dim']}: thin={row['thin']}, n={row['n_thinned']}, "f"stat={row['ks_stat']:.4f}, p={row['p_value']:.3f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()