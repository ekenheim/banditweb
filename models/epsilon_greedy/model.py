"""
bandit_runtime/epsilon_greedy.py — Epsilon-Greedy policy
─────────────────────────────────────────────────────────
Explores uniformly at random with probability epsilon,
exploits the empirically best arm otherwise.
"""
from __future__ import annotations

import numpy as np
from mlserver.settings import ModelSettings
from bandit_runtime._base import BanditBase


class EpsilonGreedyModel(BanditBase):

    MLFLOW_EXPERIMENT = "bandit-epsilon-greedy"

    def _init_state(self, params: dict) -> None:
        self.epsilon = float(params.get("epsilon", 0.1))
        self.counts = np.zeros(self.n_arms, dtype=np.int64)   # pulls per arm
        self.values = np.zeros(self.n_arms, dtype=np.float64)  # mean reward per arm

    def _select_arm(self, context=None) -> int:
        # context is ignored — non-contextual policy
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.values))

    def _update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental mean update
        self.values[arm] += (reward - self.values[arm]) / n

    def state_dict(self) -> dict:
        return {
            f"value_{i}": float(self.values[i]) for i in range(self.n_arms)
        } | {
            f"count_{i}": int(self.counts[i]) for i in range(self.n_arms)
        }
