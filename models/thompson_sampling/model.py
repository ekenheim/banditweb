"""
bandit_runtime/thompson_sampling.py — Thompson Sampling policy
──────────────────────────────────────────────────────────────
Maintains a Beta(alpha, beta) posterior for each arm's
Bernoulli reward probability.  At each step, samples
theta_a ~ Beta(alpha_a, beta_a) for each arm and selects
argmax(theta).

Assumes binary rewards {0, 1}.  Non-binary rewards are
thresholded at 0.5 for the update step.
"""
from __future__ import annotations

import numpy as np
from bandit_runtime._base import BanditBase


class ThompsonSamplingModel(BanditBase):

    MLFLOW_EXPERIMENT = "bandit-thompson-sampling"

    def _init_state(self, params: dict) -> None:
        # Prior: Beta(1, 1) = Uniform[0,1]
        self.alpha = np.ones(self.n_arms, dtype=np.float64)  # successes + 1
        self.beta  = np.ones(self.n_arms, dtype=np.float64)  # failures  + 1

    def _select_arm(self, context=None) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def _update(self, arm: int, reward: float) -> None:
        # Bernoulli update: threshold non-binary rewards at 0.5
        success = 1 if reward >= 0.5 else 0
        self.alpha[arm] += success
        self.beta[arm]  += 1 - success

    def state_dict(self) -> dict:
        # Report posterior mean = alpha / (alpha + beta)
        posterior_mean = self.alpha / (self.alpha + self.beta)
        return {
            f"value_{i}":         float(posterior_mean[i]) for i in range(self.n_arms)
        } | {
            f"alpha_{i}":         float(self.alpha[i])     for i in range(self.n_arms)
        } | {
            f"beta_{i}":          float(self.beta[i])      for i in range(self.n_arms)
        }
