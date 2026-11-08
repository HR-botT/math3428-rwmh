"""Generate all figures and tables for the report.

Every figure is rendered at a generous size with matplotlib's constrained
layout engine, which reserves space for titles, axis labels, tick labels
and legends so that nothing overlaps. Sizes (in inches, at 120 dpi):

  single-panel    11 x 7   (1320 x 840 px)
  two-panel       16 x 7   (1920 x 840 px)
  three-panel     18 x 6   (2160 x 720 px)
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(OUT, "tables"); os.makedirs(TAB, exist_ok=True)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import RandomWalkMH
from targets import gaussian_log_target, identity_cov, ar1_cov
from diagnostics import acf, ess

# Global style: bigger fonts, thick legend frames, non-overlapping titles.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.30,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# --------------------------------------------------------------- data ----
sim = pd.read_csv(os.path.join(OUT, "results_simulation_summary.csv"))
base = pd.read_csv(os.path.join(OUT, "results_baseline_summary.csv"))
bud = pd.read_csv(os.path.join(OUT, "results_budget_summary.csv"))


def _save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    print(f"  wrote {name}")


# -------------------------------------------------------- fig1: acc/scale --
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for d, grp in sim.groupby("d"):
    ax.plot(grp["rel_scale"], grp["acc_rate"], marker="o", markersize=7,
            linewidth=1.8, label=f"d = {d}")
ax.axhline(0.234, color="k", ls="--", lw=1.4,
           label="0.234 (large-d target)")
ax.axhline(0.44, color="gray", ls=":", lw=1.4,
           label="0.44 (d = 1 target)")
ax.set_xscale("log")
ax.set_xlabel(r"proposal scale $/\,\sigma^\star$")
ax.set_ylabel("acceptance rate")
ax.set_title("Acceptance rate vs proposal scale")
ax.legend(loc="upper right", framealpha=0.95)
_save(fig, "fig1_acc_vs_scale.png")


# -------------------------------------------------------- fig2: mse/scale --
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for d, grp in sim.groupby("d"):
    ax.errorbar(grp["rel_scale"], grp["mse"], yerr=grp["mse_se"],
                marker="o", markersize=7, capsize=4, linewidth=1.6,
                label=f"d = {d}")
ax.axvline(1.0, color="k", ls="--", lw=1.4,
           label=r"$\sigma = \sigma^\star$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"proposal scale $/\,\sigma^\star$")
ax.set_ylabel(r"MSE of $\hat\mu$")
ax.set_title("MSE vs proposal scale (log-log)")
ax.legend(loc="upper center", ncol=2, framealpha=0.95)
_save(fig, "fig2_mse_vs_scale.png")


# -------------------------------------------- fig3: mse vs acceptance -----
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for d, grp in sim.groupby("d"):
    g = grp.sort_values("acc_rate")
    ax.plot(g["acc_rate"], g["mse"], marker="o", markersize=7,
            linewidth=1.6, label=f"d = {d}")
ax.axvline(0.234, color="k", ls="--", lw=1.4, label="0.234")
ax.axvline(0.44, color="gray", ls=":", lw=1.4, label="0.44")
ax.set_yscale("log")
ax.set_xlabel("acceptance rate")
ax.set_ylabel(r"MSE of $\hat\mu$")
ax.set_title("MSE vs acceptance rate: U-shape")
ax.legend(loc="upper center", ncol=2, framealpha=0.95)
_save(fig, "fig3_mse_vs_accept.png")


# -------------------------------------------------------- fig4: ess/scale --
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for d, grp in sim.groupby("d"):
    ax.plot(grp["rel_scale"], grp["ess_mean"], marker="o", markersize=7,
            linewidth=1.8, label=f"d = {d}")
ax.axvline(1.0, color="k", ls="--", lw=1.4,
           label=r"$\sigma = \sigma^\star$")
ax.set_xscale("log")
ax.set_xlabel(r"proposal scale $/\,\sigma^\star$")
ax.set_ylabel("mean ESS per dimension")
ax.set_title("Effective sample size vs proposal scale")
ax.legend(loc="upper right", framealpha=0.95)
_save(fig, "fig4_ess_vs_scale.png")


# -------------------------------------------- fig5+6: trace & acf (d=5) ---
d = 5
log_t = gaussian_log_target(np.zeros(d), identity_cov(d))
sig = 2.38 / np.sqrt(d)
scales = {"small (0.2x)": 0.2 * sig,
          "optimal (1x)": sig,
          "large (5x)": 5.0 * sig}
chains = {}
for name, s in scales.items():
    rng = np.random.default_rng(42)
    mh = RandomWalkMH(log_t, d, s, rng)
    ch, acc = mh.sample(np.zeros(d), n_iter=1000, burn_in=500)
    chains[name] = (ch[:, 0], acc, s)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True,
                         constrained_layout=True)
for ax, (name, (chain, acc, s)) in zip(axes, chains.items()):
    ax.plot(chain, lw=0.8)
    ax.set_title(f"{name}\nacc = {acc:.2f}", fontsize=13)
    ax.set_xlabel("iteration")
axes[0].set_ylabel("x (dim 0)")
fig.suptitle("Trace plots (d = 5)", fontsize=15)
_save(fig, "fig5_trace.png")

fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for name, (chain, acc, s) in chains.items():
    rho = acf(chain, max_lag=100)
    if np.all(np.isnan(rho)):
        continue
    ax.plot(rho, marker="o", markersize=3, linewidth=1.5,
            label=f"{name}, acc = {acc:.2f}")
ax.axhline(0, color="k", lw=0.9)
ax.set_xlabel("lag"); ax.set_ylabel("ACF")
ax.set_title("Autocorrelation of a coordinate (d = 5)")
ax.legend(loc="upper right", framealpha=0.95)
_save(fig, "fig6_acf.png")


# -------------------------------------------------------- fig7: budget ----
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
ax.errorbar(bud["n_iter"], bud["mse"], yerr=bud["mse_se"],
            marker="o", markersize=8, capsize=5, linewidth=1.8,
            label="RWMH (optimal)")
ref = bud["mse"].iloc[-1] * bud["n_iter"].iloc[-1] / bud["n_iter"]
ax.plot(bud["n_iter"], ref, "k--", lw=1.4, label=r"$C/N$ reference")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("iterations N"); ax.set_ylabel(r"MSE of $\hat\mu$")
ax.set_title("MSE vs computational budget (d = 20)")
ax.legend(loc="upper right", framealpha=0.95)
_save(fig, "fig7_budget.png")


# ------------------------------------------------- fig8: baseline --------
fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
metrics = [("mse", "MSE of mean (log scale)"),
           ("runtime_s", "wall-clock seconds (log scale)")]
for ax, (col, label) in zip(axes, metrics):
    for method, grp in base.groupby("method"):
        ax.plot(grp["d"], grp[col], marker="o", markersize=9, linewidth=1.9,
                label=method)
    ax.set_xlabel("dimension d"); ax.set_ylabel(label)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.legend(loc="best", framealpha=0.95)
axes[0].set_title("Accuracy")
axes[1].set_title("Runtime")
fig.suptitle("RWMH(optimal) vs i.i.d. baseline", fontsize=15)
_save(fig, "fig8_baseline.png")


# ---------------------------------------- fig9: robustness (correlated) --
d_corr = 10
rho = 0.8
log_corr = gaussian_log_target(np.zeros(d_corr), ar1_cov(d_corr, rho))
sig_c = 2.38 / np.sqrt(d_corr)
rel = np.geomspace(0.2, 5.0, 10)
mse_vals = []
for r in rel:
    ms = []
    for rep in range(40):
        rng = np.random.default_rng(1_000 * rep + int(r * 100))
        mh = RandomWalkMH(log_corr, d_corr, r * sig_c, rng)
        ch, _ = mh.sample(np.zeros(d_corr), n_iter=1500, burn_in=500)
        ms.append(np.mean(ch.mean(axis=0) ** 2))
    mse_vals.append(np.mean(ms))
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
ax.plot(rel, mse_vals, marker="o", markersize=8, linewidth=1.9)
ax.axvline(1.0, color="k", ls="--", lw=1.4,
           label=r"$\sigma = \sigma^\star$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"proposal scale $/\,\sigma^\star$")
ax.set_ylabel(r"MSE of $\hat\mu$")
ax.set_title(f"Robustness: AR(1) target, d = {d_corr}, rho = {rho}")
ax.legend(loc="upper right", framealpha=0.95)
_save(fig, "fig9_robust_corr.png")


# -------------------------------------------------------- tables ----------
# Table 1
rows = []
for d, grp in sim.groupby("d"):
    best = grp.loc[grp["mse"].idxmin()]
    sigma_star = 2.38 / np.sqrt(d)
    rows.append({
        "d": d, "sigma_star_theory": sigma_star,
        "scale_at_min_mse": best["scale"],
        "scale_ratio": best["scale"] / sigma_star,
        "accept_at_min_mse": best["acc_rate"],
        "mse_at_min": best["mse"], "ess_at_min": best["ess_mean"],
    })
pd.DataFrame(rows).to_csv(os.path.join(TAB, "table1_optimal_scale.csv"),
                          index=False)

# Table 2
base.to_csv(os.path.join(TAB, "table2_baseline.csv"), index=False)

# Table 3
try:
    with open(os.path.join(OUT, "validation.json")) as f:
        v = json.load(f)
    rows = []
    for j, (m, se) in enumerate(zip(v["sample_mean"], v["mcse_of_mean"])):
        rows.append({"dim": j, "posterior_mean_est": m, "mcse": se,
                     "variance_est": v["sample_var_diag"][j],
                     "ks_stat": v["ks_tests"][j]["ks_stat"],
                     "ks_p": v["ks_tests"][j]["p_value"]})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "table3_validation.csv"),
                              index=False)
except FileNotFoundError:
    print("validation.json not found; skipping table 3")

print("Figures and tables written to", OUT)