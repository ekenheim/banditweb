"""
bandit_runtime/linucb.py — LinUCB (disjoint) contextual bandit
───────────────────────────────────────────────────────────────
Per-arm linear model: expected reward = x^T θ_a
Confidence bound:     p_a(x) = x^T θ_a + alpha * sqrt(x^T A_a^{-1} x)

Context vector (4-dim, supplied by frontend):
  [time_bucket, user_segment, recency_score, frequency_score]
  All values normalised to [0, 1] by the frontend before sending.

References: Li et al. 2010 — "A Contextual-Bandit Approach to
Personalized News Article Recommendation"
"""
from __future__ import annotations

import numpy as np
from bandit_runtime._base import BanditBase

DEFAULT_CONTEXT_DIM = 4


class LinUCBModel(BanditBase):

    MLFLOW_EXPERIMENT = "bandit-linucb"

    def _init_state(self, params: dict) -> None:
        self.alpha = float(params.get("alpha", 1.0))
        self.d = int(params.get("context_dim", DEFAULT_CONTEXT_DIM))

        # Per-arm ridge matrices and reward vectors
        # A_a = I_d  (ridge regularisation)
        # b_a = 0
        self.A = [np.eye(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]

        # Cache last context for the reward update
        self._last_context: dict[int, np.ndarray] = {}

    def _select_arm(self, context: np.ndarray | None) -> int:
        if context is None:
            # Degrade gracefully: uniform random if no context provided
            return int(np.random.randint(self.n_arms))

        x = context.reshape(self.d, 1)
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta  = A_inv @ self.b[a]
            p[a]   = theta @ x.flatten() + self.alpha * np.sqrt(
                x.T @ A_inv @ x
            )
            # Store context so we can update on reward
            self._last_context[a] = context.copy()

        return int(np.argmax(p))

    def _update(self, arm: int, reward: float) -> None:
        x = self._last_context.get(arm)
        if x is None:
            return  # No context stored — skip update
        x = x.reshape(self.d, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x.flatten()

    def state_dict(self) -> dict:
        state = {}
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta  = A_inv @ self.b[a]
            state[f"value_{a}"] = float(np.mean(theta))  # scalar summary
        return state
