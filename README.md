# Project: Random-Walk Metropolis–Hastings and Proposal Scale

MATH 3428 Statistical Computing — Fall 2026–27

**Group:** [MAI, Qinyuan (21153159)], [SIU, Tsz Cheung (SID)], [LEE, Hoi KIU (SID)]

**Question.** How does proposal scale affect the mixing and estimation accuracy
of a Metropolis–Hastings sampler, and how does the classical optimal-scaling
result (σ ∝ 2.38/√d, acceptance ≈ 0.234 for large d) hold across dimensions
and computational budgets?

## Summary

Implements a random-walk Metropolis–Hastings (RWMH) sampler from scratch and
studies it on Gaussian targets in dimensions d ∈ {1, 5, 20}, sweeping the
proposal scale over 12 log-spaced values per dimension with 100 independent
replications. The implementation is validated against analytically known
moments; performance is benchmarked against an i.i.d. baseline and studied as
a function of iteration count. Full write-up in `report.pdf`; source in
`report.ipynb`.

## What is implemented myself

- **Central algorithm:** `code/algorithm.py` — RWMH sampler.
- **Diagnostics:** `code/diagnostics.py` — autocovariance (FFT), ACF, and
  effective sample size (initial positive-sequence estimator).
- **Experiments:** validation, simulation study, baseline comparison, budget
  study, correlated-target robustness check, and all figures/tables.

Third-party packages (`numpy`, `scipy`, `pandas`, `matplotlib`) are used only
for random number generation, linear algebra, data handling, plotting, and
reference statistical tests — all permitted under §4 of the brief. **No MCMC
package is used to run the sampler.**

## Requirements

- Python ≥ 3.9
- Packages: `numpy`, `scipy`, `pandas`, `matplotlib`
- For rebuilding the report: `pandoc`, a LaTeX engine with `xelatex`
  (e.g. MiKTeX or TeX Live), and a working PowerShell environment

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## How to run

From the project root:

```bash
python code/run_all.py
```

Runs validation, the main simulation study, baseline comparison, budget study,
and figure generation in sequence.

**Runtime:** approximately **3 minutes** on a laptop with default settings
(12 scales × 3 dimensions × 100 replications at N = 1500). To reduce runtime,
lower `REPS` and `N_SCALES` in `code/simulation_study.py`; the qualitative
conclusions are robust to this.

## Report sources and build

The report has **one hand-edited source**: `report.ipynb`. The Markdown file
`report.md` and the final PDF `report.pdf` are both **generated** from the
notebook — they should not be edited by hand.

Pipeline: `report.ipynb` → `extract_report.py` → `report.md` → `pandoc` →
`report.pdf`.

### One-command build (Windows PowerShell)

From the project root:

```powershell
.\build.ps1
```

If PowerShell blocks script execution, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

The script checks that `xelatex` and `report.ipynb` are available, extracts
`report.md` from the last markdown cell of `report.ipynb`, then renders
`report.pdf` via pandoc.

### Manual build

```powershell
# 1. Extract report.md from the last markdown cell of report.ipynb
python extract_report.py

# 2. Render the PDF via pandoc + xelatex
pandoc report.md -o report.pdf --pdf-engine=xelatex `
    -V geometry:margin=1.5cm -V fontsize=10pt -V linestretch=1.05 `
    --include-in-header=header.tex
```

Both `report.md` and `report.pdf` are committed, so rebuilding is optional
for marking; it is included so the source can be regenerated from
`report.ipynb`.

## What the pipeline produces

| File | Contents |
|------|----------|
| `outputs/validation.json` | Moment recovery, MCSEs, raw and thinned KS tests on N(0, I₅) |
| `outputs/results_simulation.csv` + `_summary.csv` | Main scale × dimension study (raw runs + summaries) |
| `outputs/results_baseline.csv` + `_summary.csv` | RWMH(optimal) vs i.i.d. baseline |
| `outputs/results_budget.csv` + `_summary.csv` | MSE vs iteration count at optimal scale |
| `outputs/figures/fig1_acc_vs_scale.png` … `fig9_robust_corr.png` | All figures referenced in the report |
| `outputs/tables/table1_optimal_scale.csv` … `table3_validation.csv` | All tables referenced in the report |

All output artefacts are already committed under `outputs/`, so results can be
inspected without re-running the pipeline.

## File map

| Path | Purpose |
|------|---------|
| `code/algorithm.py` | RWMH sampler (central self-implemented algorithm) |
| `code/targets.py` | Gaussian log-densities (identity and AR(1) covariance) |
| `code/diagnostics.py` | Autocovariance, ACF, ESS (self-implemented) |
| `code/validation.py` | Recovery of known moments; raw and thinned KS tests |
| `code/simulation_study.py` | Main experiment: scale × dimension |
| `code/baseline_comparison.py` | RWMH(optimal) vs i.i.d. baseline |
| `code/budget_study.py` | MSE vs iterations at optimal scale |
| `code/make_figures.py` | All figures and tables |
| `code/run_all.py` | Runs the full pipeline in order |
| `report.ipynb` | **Report source (Jupyter Notebook; hand-edited)** |
| `extract_report.py` | Extracts the last markdown cell of `report.ipynb` → `report.md` |
| `build.ps1` | One-command build: extract → pandoc → `report.pdf` |
| `header.tex` | LaTeX header included by pandoc |
| `report.md` | Generated Markdown source (pandoc input) |
| `report.pdf` | Final 6–8 page report |
| `requirements.txt` | Python dependencies |
| `environment/` | Exact version snapshot (see below) |
| `outputs/` | All results (figures, tables, CSVs, JSON) |

**Data:** all experiments use synthetic data generated by the scripts in
`code/`. No external data files are required or used.

## Reproducibility

- **Seeds:** every script uses a fixed `BASE_SEED` (defined at the top of each
  file). Re-running the pipeline from a clean checkout reproduces all numbers
  and figures in the report.
- **Environment:** exact package versions can be captured with:

  ```bash
  pip freeze > environment/requirements_lock.txt
  python -c "import sys, scipy, numpy, pandas, matplotlib; \
    print(sys.version); \
    print(scipy.__version__, numpy.__version__, pandas.__version__, matplotlib.__version__)" \
    > environment/sessionInfo.txt
  ```

- **Rebuilding the report:** edit and save `report.ipynb`, then run
  `.\build.ps1` (or the manual build above). Requires `pandoc`, `xelatex`,
  and the Python packages listed in `requirements.txt`.

## Contribution statement

[TODO: fill in per-member contributions before submission.]