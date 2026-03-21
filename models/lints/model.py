"""
bandit_runtime/lints.py — Linear Thompson Sampling (contextual)
───────────────────────────────────────────────────────────────
Contextual bandit that combines Thompson Sampling exploration
with a linear reward model (like LinUCB).

Instead of using a deterministic upper confidence bound, LinTS
samples the weight vector from its posterior:

    theta ~ N(theta_hat, v^2 * A^{-1})
    score  = x^T theta

This gives Bayesian exploration: arms with uncertain weight
estimates are explored naturally through posterior sampling.

Context vector (4-dim, supplied by frontend):
  [time_bucket, user_segment, recency_score, frequency_score]

Reference: Agrawal & Goyal 2013 — "Thompson Sampling for
Contextual Bandits with Linear Payoffs"
"""
from __future__ import annotations

import numpy as np
from bandit_runtime._base import BanditBase

DEFAULT_CONTEXT_DIM = 4


class LinTSModel(BanditBase):

    MLFLOW_EXPERIMENT = "bandit-lints"

    def _init_state(self, params: dict) -> None:
        self.v = float(params.get("v", 1.0))
        self.d = int(params.get("context_dim", DEFAULT_CONTEXT_DIM))

        self.A = [np.eye(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]

        self._last_context: dict[int, np.ndarray] = {}

    def _select_arm(self, context: np.ndarray | None) -> int:
        if context is None:
            return int(np.random.randint(self.n_arms))

        x = context.flatten()
        scores = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            # Sample from posterior: theta ~ N(theta_hat, v^2 * A_inv)
            theta_sample = np.random.multivariate_normal(theta_hat, self.v ** 2 * A_inv)
            scores[a] = x @ theta_sample
            self._last_context[a] = context.copy()

        return int(np.argmax(scores))

    def _update(self, arm: int, reward: float) -> None:
        x = self._last_context.get(arm)
        if x is None:
            return
        x = x.reshape(self.d, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x.flatten()

    def state_dict(self) -> dict:
        state = {}
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            state[f"value_{a}"] = float(np.mean(theta))
        return state
