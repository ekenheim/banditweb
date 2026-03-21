"""
analysis/app.py — FastAPI service for PyMC Bayesian deep dive.

Provides endpoints for posterior analysis, convergence diagnostics,
and model comparison using PyMC and ArviZ.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import (
    build_beta_model,
    build_linear_model,
    compute_diagnostics,
    compute_model_comparison,
)

matplotlib.use("Agg")

app = FastAPI(title="Bandit Bayesian Analysis", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────


class BetaAnalysisRequest(BaseModel):
    alpha: list[float]
    beta: list[float]
    arm_labels: Optional[list[str]] = None
    n_samples: int = 2000


class LinearAnalysisRequest(BaseModel):
    A_matrices: list[list[list[float]]]
    b_vectors: list[list[float]]
    arm_labels: Optional[list[str]] = None
    n_samples: int = 2000


class CompareRequest(BaseModel):
    states: list[dict]


# ── Helpers ────────────────────────────────────────────────────────────────


def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _style_axes_dark(fig):
    """Apply dark theme to all axes in a figure."""
    for ax in fig.get_axes():
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
    fig.patch.set_facecolor("#0d1117")
    fig.tight_layout()


def make_trace_plot(idata: az.InferenceData, var_names: list[str]) -> str:
    """Generate a trace plot and return as base64 PNG."""
    axes = az.plot_trace(idata, var_names=var_names)
    fig = axes.ravel()[0].get_figure()
    _style_axes_dark(fig)
    return fig_to_base64(fig)


def make_posterior_plot(idata: az.InferenceData, var_names: list[str]) -> str:
    """Generate a posterior distribution plot and return as base64 PNG."""
    axes = az.plot_posterior(idata, var_names=var_names)
    axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]
    fig = axes_flat[0].get_figure()
    _style_axes_dark(fig)
    return fig_to_base64(fig)


def make_ppc_plot(idata: az.InferenceData) -> str | None:
    """Generate a posterior predictive check plot if available."""
    if not hasattr(idata, "posterior_predictive"):
        return None

    try:
        axes = az.plot_ppc(idata)
        axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]
        fig = axes_flat[0].get_figure()
        _style_axes_dark(fig)
        for ax in fig.get_axes():
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor("#161b22")
                legend.get_frame().set_edgecolor("#30363d")
                for text in legend.get_texts():
                    text.set_color("#c9d1d9")
        return fig_to_base64(fig)
    except Exception:
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.post("/analyze/beta")
async def analyze_beta(req: BetaAnalysisRequest):
    """
    Analyze Beta-Bernoulli bandit state.
    Used for: Epsilon-Greedy, UCB, Thompson Sampling, Bayesian UCB.
    """
    if len(req.alpha) != len(req.beta):
        raise HTTPException(400, "alpha and beta must have the same length")

    n_arms = len(req.alpha)
    idata = build_beta_model(req.alpha, req.beta, n_samples=req.n_samples)

    var_names = ["theta"]
    diagnostics = compute_diagnostics(idata)

    return {
        "diagnostics": diagnostics,
        "plots": {
            "trace": make_trace_plot(idata, var_names),
            "posterior": make_posterior_plot(idata, var_names),
            "ppc": make_ppc_plot(idata),
        },
        "n_arms": n_arms,
        "arm_labels": req.arm_labels or [f"Arm {chr(65 + i)}" for i in range(n_arms)],
    }


@app.post("/analyze/linear")
async def analyze_linear(req: LinearAnalysisRequest):
    """
    Analyze linear contextual bandit state.
    Used for: LinUCB, LinTS.
    """
    n_arms = len(req.A_matrices)
    idata = build_linear_model(req.A_matrices, req.b_vectors, n_samples=req.n_samples)

    var_names = [f"theta_{a}" for a in range(n_arms)]
    diagnostics = compute_diagnostics(idata)

    return {
        "diagnostics": diagnostics,
        "plots": {
            "trace": make_trace_plot(idata, var_names[:3]),  # Limit to first 3 arms for readability
            "posterior": make_posterior_plot(
                idata,
                [f"expected_reward_{a}" for a in range(n_arms)],
            ),
        },
        "n_arms": n_arms,
        "arm_labels": req.arm_labels or [f"Arm {chr(65 + i)}" for i in range(n_arms)],
    }


@app.post("/analyze/compare")
async def analyze_compare(req: CompareRequest):
    """
    Compare multiple policies via LOO-CV.
    Only works for Beta-based policies.
    """
    result = compute_model_comparison(req.states)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
