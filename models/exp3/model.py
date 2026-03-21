"""
bandit_runtime/exp3.py — EXP3 (Exponential-weight algorithm for
Exploration and Exploitation)
─────────────────────────────────────────────────────────────────
Adversarial bandit algorithm that maintains probability weights
over arms.  Unlike stochastic algorithms, EXP3 makes no
assumptions about reward distributions — it provides guarantees
even when rewards are chosen by an adversary.

Mixed strategy:
    p_i = (1 - gamma) * w_i / sum(w) + gamma / K

Update (importance-weighted):
    estimated_reward = reward / p[arm]
    w[arm] *= exp(gamma * estimated_reward / K)

Reference: Auer et al. 2002 — "The Nonstochastic Multiarmed
Bandit Problem"
"""
from __future__ import annotations

import numpy as np
from bandit_runtime._base import BanditBase


class EXP3Model(BanditBase):

    MLFLOW_EXPERIMENT = "bandit-exp3"

    def _init_state(self, params: dict) -> None:
        self.gamma = float(params.get("gamma", 0.1))
        self.weights = np.ones(self.n_arms, dtype=np.float64)
        self._probabilities = np.ones(self.n_arms, dtype=np.float64) / self.n_arms
        self._counts = np.zeros(self.n_arms, dtype=np.int64)
        self._cum_rewards = np.zeros(self.n_arms, dtype=np.float64)

    def _compute_probabilities(self) -> np.ndarray:
        w_sum = np.sum(self.weights)
        p = (1 - self.gamma) * self.weights / w_sum + self.gamma / self.n_arms
        return p

    def _select_arm(self, context=None) -> int:
        self._probabilities = self._compute_probabilities()
        return int(np.random.choice(self.n_arms, p=self._probabilities))

    def _update(self, arm: int, reward: float) -> None:
        self._counts[arm] += 1
        self._cum_rewards[arm] += reward
        p = self._probabilities[arm]
        estimated_reward = reward / max(p, 1e-10)
        self.weights[arm] *= np.exp(self.gamma * estimated_reward / self.n_arms)
        # Normalize weights to prevent overflow
        self.weights /= np.max(self.weights)

    def state_dict(self) -> dict:
        probs = self._compute_probabilities()
        mean_rewards = np.where(
            self._counts > 0,
            self._cum_rewards / self._counts,
            0.0,
        )
        return {
            f"value_{i}": float(mean_rewards[i]) for i in range(self.n_arms)
        } | {
            f"weight_{i}": float(self.weights[i]) for i in range(self.n_arms)
        } | {
            f"prob_{i}": float(probs[i]) for i in range(self.n_arms)
        } | {
            f"count_{i}": int(self._counts[i]) for i in range(self.n_arms)
        }
