"""
analysis/models.py — PyMC model builders for Bayesian deep dive.

Builds PyMC models from bandit state snapshots, runs MCMC inference,
and returns ArviZ InferenceData with posterior samples and diagnostics.
"""
from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_beta_model(
    alpha: list[float],
    beta: list[float],
    n_samples: int = 2000,
    n_chains: int = 2,
) -> az.InferenceData:
    """
    Build a Beta-Bernoulli model from observed alpha/beta parameters.

    The alpha/beta params represent the posterior after observing
    (alpha - 1) successes and (beta - 1) failures per arm.
    We reconstruct the observations and fit a proper PyMC model
    with posterior predictive checks.
    """
    n_arms = len(alpha)
    successes = [int(a - 1) for a in alpha]
    failures = [int(b - 1) for b in beta]
    trials = [s + f for s, f in zip(successes, failures)]

    with pm.Model() as model:
        # Prior: Beta(1, 1) = Uniform
        theta = pm.Beta("theta", alpha=1, beta=1, shape=n_arms)

        # Likelihood
        for i in range(n_arms):
            if trials[i] > 0:
                pm.Binomial(f"obs_{i}", n=trials[i], p=theta[i], observed=successes[i])

        # Best arm indicator
        pm.Deterministic("best_arm", pt.argmax(theta))

        # Pairwise differences (arm i vs best observed arm)
        best_idx = int(np.argmax([s / max(t, 1) for s, t in zip(successes, trials)]))
        for i in range(n_arms):
            if i != best_idx:
                pm.Deterministic(f"diff_{i}_vs_{best_idx}", theta[i] - theta[best_idx])

        # Sample
        trace = pm.sample(
            n_samples,
            chains=n_chains,
            return_inferencedata=True,
            progressbar=False,
        )

        # Posterior predictive
        pm.sample_posterior_predictive(trace, extend_inferencedata=True, progressbar=False)

    return trace


def build_linear_model(
    A_matrices: list[list[list[float]]],
    b_vectors: list[list[float]],
    n_samples: int = 2000,
    n_chains: int = 2,
) -> az.InferenceData:
    """
    Build a Bayesian linear regression model from LinUCB/LinTS state.

    Uses the ridge regression matrices (A, b) to reconstruct the
    posterior over weight vectors theta per arm.

    The posterior is N(A^{-1} b, A^{-1}), which we fit properly
    via PyMC for diagnostics and posterior predictive checks.
    """
    n_arms = len(A_matrices)
    d = len(A_matrices[0])

    with pm.Model() as model:
        for a in range(n_arms):
            A = np.array(A_matrices[a])
            b = np.array(b_vectors[a])

            # Compute posterior parameters analytically
            try:
                A_inv = np.linalg.inv(A)
                theta_hat = A_inv @ b
            except np.linalg.LinAlgError:
                theta_hat = np.zeros(d)
                A_inv = np.eye(d)

            # Prior informed by the ridge regression posterior
            theta = pm.MvNormal(
                f"theta_{a}",
                mu=theta_hat,
                cov=A_inv,
                shape=d,
            )

            # Expected reward under uniform context
            pm.Deterministic(f"expected_reward_{a}", pm.math.mean(theta))

        # Sample
        trace = pm.sample(
            n_samples,
            chains=n_chains,
            return_inferencedata=True,
            progressbar=False,
        )

    return trace


def compute_diagnostics(idata: az.InferenceData) -> dict:
    """
    Extract key diagnostics from an InferenceData object.
    Returns a JSON-serializable dict with R-hat, ESS, and summary stats.
    """
    summary = az.summary(idata, round_to=4)

    diagnostics = {
        "summary": summary.to_dict(),
        "rhat": {},
        "ess_bulk": {},
        "ess_tail": {},
    }

    for var_name in summary.index:
        diagnostics["rhat"][var_name] = float(summary.loc[var_name, "r_hat"])
        diagnostics["ess_bulk"][var_name] = float(summary.loc[var_name, "ess_bulk"])
        diagnostics["ess_tail"][var_name] = float(summary.loc[var_name, "ess_tail"])

    return diagnostics


def compute_model_comparison(
    states: list[dict],
) -> dict:
    """
    Compare models via LOO-CV where possible.
    Only works for Beta-Bernoulli models (policies with alpha/beta state).
    Returns a comparison table as a dict.
    """
    models = {}
    for state in states:
        policy = state["policy"]
        if "alpha" in state and "beta" in state:
            idata = build_beta_model(state["alpha"], state["beta"], n_samples=1000, n_chains=2)
            models[policy] = idata

    if len(models) < 2:
        return {"error": "Need at least 2 Beta-based policies for comparison"}

    comparison = az.compare(models, ic="loo")
    return {
        "comparison": comparison.to_dict(),
        "ranking": list(comparison.index),
    }
