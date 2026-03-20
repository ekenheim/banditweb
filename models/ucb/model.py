"""
bandit_runtime/ucb.py — UCB1 policy
─────────────────────────────────────────────────────────
Selects the arm with the highest upper confidence bound:
    UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))

where Q(a) is the estimated mean reward, t is the total
number of pulls, and N(a) is pulls for arm a.

Unpulled arms have UCB = +inf and are always tried first.
"""
from __future__ import annotations

import numpy as np
from bandit_runtime._base import BanditBase


class UCBModel(BanditBase):

    MLFLOW_EXPERIMENT = "bandit-ucb"

    def _init_state(self, params: dict) -> None:
        self.c = float(params.get("c", 2.0))
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.values = np.zeros(self.n_arms, dtype=np.float64)
        self.total_pulls = 0

    def _select_arm(self, context=None) -> int:
        # Always pull unpulled arms first
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])

        t = self.total_pulls
        bonus = self.c * np.sqrt(np.log(t) / self.counts)
        ucb_values = self.values + bonus
        return int(np.argmax(ucb_values))

    def _update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.total_pulls += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def state_dict(self) -> dict:
        t = max(self.total_pulls, 1)
        bonus = self.c * np.sqrt(np.log(t) / np.maximum(self.counts, 1))
        ucb = self.values + bonus
        return {
            f"value_{i}": float(self.values[i]) for i in range(self.n_arms)
        } | {
            f"ucb_{i}": float(ucb[i]) for i in range(self.n_arms)
        } | {
            f"count_{i}": int(self.counts[i]) for i in range(self.n_arms)
        }
