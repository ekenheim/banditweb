"""
ucb.py — UCB1 bandit policy (self-contained)
─────────────────────────────────────────────
Selects the arm with the highest upper confidence bound:
    UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
"""
from __future__ import annotations

import os
import threading
import time
from abc import abstractmethod

import mlflow
import numpy as np
from mlserver import MLModel
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
)


# ── Base class (inlined) ──────────────────────────────────────────────────

class BanditBase(MLModel):
    N_ARMS: int = 5
    MLFLOW_EXPERIMENT: str = "bandit-default"

    async def load(self) -> bool:
        params = (self.settings.parameters.extra if self.settings.parameters else {}) or {}
        self.n_arms = int(params.get("n_arms", self.N_ARMS))
        self.mlflow_experiment = params.get("mlflow_experiment", self.MLFLOW_EXPERIMENT)
        self._lock = threading.Lock()
        self._step = 0
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.end_run()
        self._run = mlflow.start_run(run_name=f"{self.name}-{int(time.time())}")
        self._init_state(params)
        return True

    @abstractmethod
    def _init_state(self, params: dict) -> None: ...
    @abstractmethod
    def _select_arm(self, context: np.ndarray | None) -> int: ...
    @abstractmethod
    def _update(self, arm: int, reward: float) -> None: ...
    @abstractmethod
    def state_dict(self) -> dict: ...

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        inputs = {inp.name: inp for inp in payload.inputs}
        if "reset" in inputs:
            return await self._handle_reset()
        if "reward" in inputs:
            data = np.array(inputs["reward"].data).flatten()
            return await self._handle_reward(int(data[0]), float(data[1]))
        context = None
        if "context" in inputs:
            context = np.array(inputs["context"].data, dtype=np.float64).flatten()
        return await self._handle_select(context)

    async def _handle_select(self, context) -> InferenceResponse:
        with self._lock:
            arm = self._select_arm(context)
            self._step += 1
            step = self._step
        mlflow.log_metric("selected_arm", arm, step=step)
        return InferenceResponse(model_name=self.name, outputs=[
            ResponseOutput(name="arm", shape=[1], datatype="INT32", data=[arm]),
            ResponseOutput(name="step", shape=[1], datatype="INT64", data=[step]),
        ])

    async def _handle_reward(self, arm: int, reward: float) -> InferenceResponse:
        with self._lock:
            self._update(arm, reward)
            step = self._step
            state = self.state_dict()
        mlflow.log_metrics({"reward": reward, "arm": arm, **{f"value_arm_{i}": float(state.get(f"value_{i}", 0)) for i in range(self.n_arms)}}, step=step)
        outputs = [ResponseOutput(name="step", shape=[1], datatype="INT64", data=[step])]
        for k, v in state.items():
            outputs.append(ResponseOutput(name=k, shape=[1], datatype="FP64", data=[float(v) if not isinstance(v, list) else v[0]]))
        return InferenceResponse(model_name=self.name, outputs=outputs)

    async def _handle_reset(self) -> InferenceResponse:
        mlflow.end_run()
        with self._lock:
            self._step = 0
            params = (self.settings.parameters.extra if self.settings.parameters else {}) or {}
            self._init_state(params)
        self._run = mlflow.start_run(run_name=f"{self.name}-{int(time.time())}")
        return InferenceResponse(model_name=self.name, outputs=[
            ResponseOutput(name="status", shape=[1], datatype="BYTES", data=["ok"]),
            ResponseOutput(name="step", shape=[1], datatype="INT64", data=[0]),
        ])


# ── Policy ────────────────────────────────────────────────────────────────

class UCBModel(BanditBase):

    MLFLOW_EXPERIMENT = "bandit-ucb"

    def _init_state(self, params: dict) -> None:
        self.c = float(params.get("c", 2.0))
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.values = np.zeros(self.n_arms, dtype=np.float64)
        self.total_pulls = 0

    def _select_arm(self, context=None) -> int:
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        t = self.total_pulls
        bonus = self.c * np.sqrt(np.log(t) / self.counts)
        return int(np.argmax(self.values + bonus))

    def _update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.total_pulls += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def state_dict(self) -> dict:
        t = max(self.total_pulls, 1)
        bonus = self.c * np.sqrt(np.log(t) / np.maximum(self.counts, 1))
        ucb = self.values + bonus
        return {f"value_{i}": float(self.values[i]) for i in range(self.n_arms)} | \
               {f"ucb_{i}": float(ucb[i]) for i in range(self.n_arms)} | \
               {f"count_{i}": int(self.counts[i]) for i in range(self.n_arms)}
