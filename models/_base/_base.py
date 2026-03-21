"""
bandit_runtime/_base.py
─────────────────────────────────────────────────────────────────────────────
Base class for all bandit MLServer runtimes.

Handles:
  - MLflow run management (one run per reset cycle)
  - Reward posting via a secondary /predict call (data=[arm, reward])
  - Thread-safe state updates
  - Graceful /reset via a special input name
"""
from __future__ import annotations

import os
import threading
import time
from abc import abstractmethod
from typing import List

import mlflow
import numpy as np
from mlserver import MLModel
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    Parameters,
    RequestInput,
    ResponseOutput,
)


class BanditBase(MLModel):
    """
    Subclass and implement:
        _init_state()   — initialise arm counts / parameters
        _select_arm()   — pure policy logic, return int arm index
        _update(arm, reward) — update posterior / statistics
        state_dict()    — return dict of loggable state for MLflow
    """

    N_ARMS: int = 5
    MLFLOW_EXPERIMENT: str = "bandit-default"

    async def load(self) -> bool:
        params = self.settings.parameters or {}

        self.n_arms = int(params.get("n_arms", self.N_ARMS))
        self.mlflow_experiment = params.get("mlflow_experiment", self.MLFLOW_EXPERIMENT)
        self._lock = threading.Lock()
        self._step = 0

        # MLflow setup
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.end_run()
        self._run = mlflow.start_run(run_name=f"{self.name}-{int(time.time())}")

        self._init_state(params)
        return True

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _init_state(self, params: dict) -> None: ...

    @abstractmethod
    def _select_arm(self, context: np.ndarray | None) -> int: ...

    @abstractmethod
    def _update(self, arm: int, reward: float) -> None: ...

    @abstractmethod
    def state_dict(self) -> dict: ...

    # ── MLServer predict entrypoint ───────────────────────────────────────────

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """
        Two operation modes, distinguished by input name:

        1. input_name == "context"  (or omitted)
           → Select an arm. Returns {"arm": int, "step": int}

        2. input_name == "reward"
           → Accept a (arm, reward) pair and update state.
              Input data: [arm_index, reward_value]
              Returns {"step": int, **state_dict()}

        3. input_name == "reset"
           → Zero all state, start a new MLflow run.
              Returns {"status": "ok", "step": 0}
        """
        inputs = {inp.name: inp for inp in payload.inputs}

        # ── Reset ─────────────────────────────────────────────────────────────
        if "reset" in inputs:
            return await self._handle_reset()

        # ── Reward update ─────────────────────────────────────────────────────
        if "reward" in inputs:
            data = np.array(inputs["reward"].data).flatten()
            arm, reward = int(data[0]), float(data[1])
            return await self._handle_reward(arm, reward)

        # ── Arm selection ─────────────────────────────────────────────────────
        context = None
        if "context" in inputs:
            context = np.array(inputs["context"].data, dtype=np.float64).flatten()

        return await self._handle_select(context)

    # ── Handlers ─────────────────────────────────────────────────────────────

    async def _handle_select(self, context) -> InferenceResponse:
        with self._lock:
            arm = self._select_arm(context)
            self._step += 1
            step = self._step

        mlflow.log_metric("selected_arm", arm, step=step)

        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(
                    name="arm",
                    shape=[1],
                    datatype="INT32",
                    data=[arm],
                ),
                ResponseOutput(
                    name="step",
                    shape=[1],
                    datatype="INT64",
                    data=[step],
                ),
            ],
        )

    async def _handle_reward(self, arm: int, reward: float) -> InferenceResponse:
        with self._lock:
            self._update(arm, reward)
            step = self._step
            state = self.state_dict()

        mlflow.log_metrics(
            {
                "reward": reward,
                "arm": arm,
                **{f"value_arm_{i}": float(state.get(f"value_{i}", 0)) for i in range(self.n_arms)},
            },
            step=step,
        )

        outputs = [
            ResponseOutput(name="step", shape=[1], datatype="INT64", data=[step]),
        ]
        for k, v in state.items():
            outputs.append(
                ResponseOutput(name=k, shape=[1], datatype="FP64", data=[float(v) if not isinstance(v, list) else v[0]])
            )

        return InferenceResponse(model_name=self.name, outputs=outputs)

    async def _handle_reset(self) -> InferenceResponse:
        # End the current MLflow run
        mlflow.end_run()

        with self._lock:
            self._step = 0
            self._init_state(
                {k: v for k, v in (self.settings.parameters or {}).items()}
            )

        # Start a fresh run
        self._run = mlflow.start_run(run_name=f"{self.name}-{int(time.time())}")

        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(name="status", shape=[1], datatype="BYTES", data=["ok"]),
                ResponseOutput(name="step", shape=[1], datatype="INT64", data=[0]),
            ],
        )
