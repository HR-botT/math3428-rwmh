"""Target distributions (log-densities) used in the study."""
from __future__ import annotations
import numpy as np


def gaussian_log_target(mean, cov):
    """Log-density of N(mean, cov), up to nothing — fully normalised."""
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")
    prec = np.linalg.inv(cov)
    d = mean.size
    const = -0.5 * (d * np.log(2.0 * np.pi) + logdet)

    def logp(x):
        diff = x - mean
        return const - 0.5 * float(diff @ prec @ diff)

    return logp


def identity_cov(d):
    return np.eye(d)


def ar1_cov(d, rho):
    """Correlated covariance: AR(1) with lag-1 correlation rho."""
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])