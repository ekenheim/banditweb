"""
bandit_runtime/bayesian_ucb.py — Bayesian UCB policy
─────────────────────────────────────────────────────
Maintains a Beta(alpha, beta) posterior for each arm (like
Thompson Sampling), but selects arms using an upper credible
bound rather than random sampling:

    score(a) = mean(a) + c * std(a)

where mean = alpha / (alpha + beta) and std is the Beta
standard deviation.  The parameter c controls the width of
the credible interval (higher c → more exploration).

Assumes binary rewards {0, 1}.  Non-binary rewards are
thresholded at 0.5 for the update step.
"""
from __future__ import annotations

import numpy as np
from bandit_runtime._base import BanditBase


class BayesianUCBModel(BanditBase):

    MLFLOW_EXPERIMENT = "bandit-bayesian-ucb"

    def _init_state(self, params: dict) -> None:
        self.c = float(params.get("c", 3.0))
        self.alpha = np.ones(self.n_arms, dtype=np.float64)
        self.beta = np.ones(self.n_arms, dtype=np.float64)

    def _select_arm(self, context=None) -> int:
        mean = self.alpha / (self.alpha + self.beta)
        ab = self.alpha + self.beta
        std = np.sqrt(self.alpha * self.beta / (ab * ab * (ab + 1)))
        ucb = mean + self.c * std
        return int(np.argmax(ucb))

    def _update(self, arm: int, reward: float) -> None:
        success = 1 if reward >= 0.5 else 0
        self.alpha[arm] += success
        self.beta[arm] += 1 - success

    def state_dict(self) -> dict:
        mean = self.alpha / (self.alpha + self.beta)
        ab = self.alpha + self.beta
        std = np.sqrt(self.alpha * self.beta / (ab * ab * (ab + 1)))
        ucb = mean + self.c * std
        return {
            f"value_{i}": float(mean[i]) for i in range(self.n_arms)
        } | {
            f"ucb_{i}": float(ucb[i]) for i in range(self.n_arms)
        } | {
            f"alpha_{i}": float(self.alpha[i]) for i in range(self.n_arms)
        } | {
            f"beta_{i}": float(self.beta[i]) for i in range(self.n_arms)
        }
