"""
analysis/models.py — PyMC model builders for Bayesian deep dive.

Builds PyMC models from bandit state snapshots, runs MCMC inference,
and returns ArviZ InferenceData with posterior samples and diagnostics.

PyMC v5+ with PyTensor backend.
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

    This lets us:
    - Verify convergence with R-hat and ESS diagnostics
    - Generate posterior predictive samples
    - Compute P(best arm) from the full posterior
    - Measure pairwise differences between arms
    """
    n_arms = len(alpha)
    successes = [max(0, int(a - 1)) for a in alpha]
    failures = [max(0, int(b - 1)) for b in beta]
    trials = [s + f for s, f in zip(successes, failures)]

    with pm.Model() as model:
        # Prior: Beta(1, 1) = Uniform — uninformative prior
        theta = pm.Beta("theta", alpha=1, beta=1, shape=n_arms)

        # Likelihood: Binomial observations per arm
        for i in range(n_arms):
            if trials[i] > 0:
                pm.Binomial(f"obs_{i}", n=trials[i], p=theta[i], observed=successes[i])

        # Derived quantities tracked as deterministics
        # Best arm indicator (which arm has highest reward probability)
        pm.Deterministic("best_arm", pt.argmax(theta))

        # P(reward) difference between each arm and the empirically best arm
        best_idx = int(np.argmax([s / max(t, 1) for s, t in zip(successes, trials)]))
        for i in range(n_arms):
            if i != best_idx:
                pm.Deterministic(f"diff_{i}_vs_{best_idx}", theta[i] - theta[best_idx])

        # Expected loss: how much reward we lose by picking each arm vs the true best
        pm.Deterministic("expected_loss", pt.max(theta) - theta)

        # Sample posterior
        trace = pm.sample(
            n_samples,
            chains=n_chains,
            return_inferencedata=True,
            progressbar=False,
        )

        # Posterior predictive: "given what we learned, what rewards do we expect?"
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

    The analytic posterior is N(A^{-1} b, A^{-1}). We encode this as
    a PyMC model so we get proper MCMC diagnostics, posterior samples,
    and can compute derived quantities like expected reward under
    different contexts.
    """
    n_arms = len(A_matrices)
    d = len(A_matrices[0])

    with pm.Model() as model:
        expected_rewards = []

        for a in range(n_arms):
            A = np.array(A_matrices[a])
            b = np.array(b_vectors[a])

            # Compute analytic posterior parameters
            try:
                A_inv = np.linalg.inv(A)
                theta_hat = A_inv @ b
            except np.linalg.LinAlgError:
                theta_hat = np.zeros(d)
                A_inv = np.eye(d)

            # Posterior: theta_a ~ MvNormal(theta_hat, A^{-1})
            theta = pm.MvNormal(
                f"theta_{a}",
                mu=theta_hat,
                cov=A_inv,
                shape=d,
            )

            # Expected reward under uniform context [0.5, 0.5, ..., 0.5]
            # This gives a scalar summary of "how good is this arm on average"
            uniform_ctx = np.full(d, 0.5)
            er = pm.Deterministic(f"expected_reward_{a}", pt.dot(theta, uniform_ctx))
            expected_rewards.append(er)

        # Stack expected rewards to find the best arm
        er_stack = pt.stack(expected_rewards)
        pm.Deterministic("best_arm", pt.argmax(er_stack))

        # Sample posterior
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

    R-hat < 1.01 indicates convergence.
    ESS > 400 indicates sufficient effective samples.
    """
    summary = az.summary(idata, round_to=4)

    diagnostics = {
        "summary": summary.to_dict(),
        "rhat": {},
        "ess_bulk": {},
        "ess_tail": {},
    }

    for var_name in summary.index:
        row = summary.loc[var_name]
        diagnostics["rhat"][var_name] = float(row.get("r_hat", 1.0))
        diagnostics["ess_bulk"][var_name] = float(row.get("ess_bulk", 0))
        diagnostics["ess_tail"][var_name] = float(row.get("ess_tail", 0))

    return diagnostics


def compute_model_comparison(
    states: list[dict],
) -> dict:
    """
    Compare policies via LOO-CV (Leave-One-Out Cross-Validation).

    Only works for Beta-Bernoulli policies (those with alpha/beta state).
    Returns a ranking of policies by expected log pointwise predictive
    density (elpd_loo) — higher is better.
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
