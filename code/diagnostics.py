"""Autocovariance, ACF, and effective sample size (self-implemented)."""
from __future__ import annotations
import numpy as np


def autocov(x):
    """Biased autocovariance via FFT: gamma[k] = (1/n) sum (x_i - xbar)(x_{i+k}-xbar)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    xm = x - x.mean()
    f = np.fft.fft(xm, n=2 * n)
    acov = np.fft.ifft(f * np.conj(f))[:n].real / n
    return acov


def acf(x, max_lag=None):
    """Sample autocorrelation function.

    Returns a NaN array with a RuntimeWarning if the input series has zero
    variance (which happens when a chain is stuck and never moves). This keeps
    the calling code from crashing and makes the stuck chain visible in plots.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    g0 = autocov(x)[0]
    if g0 <= 0.0:
        import warnings
        warnings.warn(
            f"acf: series of length {n} has zero variance; chain appears "
            "to be stuck. Returning NaNs.",
            RuntimeWarning,
        )
        out = np.full(n, np.nan)
        return out if max_lag is None else out[: max_lag + 1]
    rho = autocov(x) / g0
    return rho if max_lag is None else rho[: max_lag + 1]


def ess(x):
    """Effective sample size, initial positive-sequence estimator."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3 or np.allclose(x, x[0]):
        return 1.0
    rho = acf(x)
    s = 0.0
    for k in range(1, n):
        if rho[k] <= 0.0:
            break
        s += rho[k]
    tau = 1.0 + 2.0 * s
    return float(n / tau)