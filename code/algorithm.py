"""Self-implemented Random-Walk Metropolis-Hastings sampler.

Central algorithm for the project. Works with any target specified via a
log-density function. Proposal: x' = x + scale * N(0, I_d).
"""
from __future__ import annotations
import numpy as np


class RandomWalkMH:
    def __init__(self, log_target, dim, scale, rng=None):
        self.log_target = log_target
        self.dim = int(dim)
        self.scale = float(scale)
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, x0, n_iter, burn_in=0, thin=1):
        """Run the chain; return (retained draws, acceptance_rate_after_burnin)."""
        x = np.asarray(x0, dtype=float).copy()
        if x.shape != (self.dim,):
            raise ValueError(f"x0 must have shape ({self.dim},), got {x.shape}")
        lp = float(self.log_target(x))
        total = burn_in + n_iter
        chain = np.empty((total, self.dim), dtype=float)
        n_accept = 0
        for i in range(total):
            prop = x + self.scale * self.rng.standard_normal(self.dim)
            lp_prop = float(self.log_target(prop))
            log_alpha = lp_prop - lp
            if np.log(self.rng.random()) < log_alpha:  # log U < log alpha
                x, lp = prop, lp_prop
                if i >= burn_in:
                    n_accept += 1
            chain[i] = x
        draws = chain[burn_in::thin]
        acc_rate = n_accept / max(n_iter, 1)
        return draws, acc_rate