"""
linucb.py — LinUCB (disjoint) contextual bandit (self-contained)
────────────────────────────────────────────────────────────────
Per-arm linear model: expected reward = x^T theta_a
Confidence bound:     p_a(x) = x^T theta_a + alpha * sqrt(x^T A_a^{-1} x)

Reference: Li et al. 2010
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

DEFAULT_CONTEXT_DIM = 4


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

class LinUCBModel(BanditBase):

    MLFLOW_EXPERIMENT = "bandit-linucb"

    def _init_state(self, params: dict) -> None:
        self.alpha = float(params.get("alpha", 1.0))
        self.d = int(params.get("context_dim", DEFAULT_CONTEXT_DIM))
        self.A = [np.eye(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]
        self._last_context: dict[int, np.ndarray] = {}

    def _select_arm(self, context: np.ndarray | None) -> int:
        if context is None:
            return int(np.random.randint(self.n_arms))
        x = context.reshape(self.d, 1)
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta @ x.flatten() + self.alpha * np.sqrt(x.T @ A_inv @ x)
            self._last_context[a] = context.copy()
        return int(np.argmax(p))

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
