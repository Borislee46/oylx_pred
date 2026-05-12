# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
from functools import lru_cache
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm, qmc

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    MONTE_CARLO_DEFAULTS,
)

_CHOLESKY_CACHE: dict[tuple, Any] = {}
_SOBOL_CACHE: dict[int, qmc.Sobol] = {}
_CACHE_LOCK = Lock()

# Shrinkage regularization: phi → phi * n_ij / (n_ij + LAMBDA)
# Low-sample pairs (n_ij < LAMBDA) have their correlation pulled toward zero.
_SHRINK_LAMBDA: float = 5.0

# Block decomposition: pairs with |phi| < COMPONENT_THRESHOLD are treated as
# independent, splitting the correlation graph into smaller connected components.
_COMPONENT_THRESHOLD: float = 0.03


@lru_cache(maxsize=128)
def _get_cholesky_decomposition(matrix_tuple: tuple) -> np.ndarray:
    n = int(np.sqrt(len(matrix_tuple)))
    matrix = np.array(matrix_tuple).reshape(n, n)

    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        fixed_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(fixed_matrix)))
        fixed_matrix_normalized = diag_inv_sqrt @ fixed_matrix @ diag_inv_sqrt
        return np.linalg.cholesky(fixed_matrix_normalized)


def _categorize_schools(
    schools: list[dict[str, Any]], correlation_matrix: pd.DataFrame
) -> tuple[list, list]:
    if correlation_matrix is None or correlation_matrix.empty:
        return [], schools.copy()

    matrix_keys = set(correlation_matrix.index)

    correlated = [
        school for school in schools if f"{school['university']} - {school['major']}" in matrix_keys
    ]
    independent = [
        school
        for school in schools
        if f"{school['university']} - {school['major']}" not in matrix_keys
    ]

    return correlated, independent


def _get_correlation_matrix(
    correlated_schools: list[dict],
    correlation_matrix: pd.DataFrame,
    pair_weight_matrix: pd.DataFrame | None = None,
) -> tuple | None:
    if not correlated_schools:
        return None

    school_keys = [f"{s['university']} - {s['major']}" for s in correlated_schools]
    sub_corr_df = correlation_matrix.loc[school_keys, school_keys]
    corr_matrix = sub_corr_df.values.astype(float)

    # ── Optimization 1: shrinkage by joint sample count ──
    if pair_weight_matrix is not None and not pair_weight_matrix.empty:
        try:
            weight_sub = pair_weight_matrix.loc[school_keys, school_keys]
            n_ij = weight_sub.values.astype(float)
            shrinkage = n_ij / (n_ij + _SHRINK_LAMBDA)
            corr_matrix = corr_matrix * shrinkage
        except KeyError:
            pass

    np.fill_diagonal(corr_matrix, 1.0)
    return tuple(np.round(corr_matrix.flatten(), 6))


def _find_correlation_components(
    corr_matrix: np.ndarray,
    threshold: float = _COMPONENT_THRESHOLD,
) -> list[np.ndarray]:
    """Split schools into connected components where |correlation| > threshold.

    Returns list of index arrays, one per component.
    """
    k = corr_matrix.shape[0]
    if k <= 1:
        return [np.arange(k)]

    adj = np.abs(corr_matrix) > threshold
    np.fill_diagonal(adj, False)

    if not adj.any():
        return [np.array([i]) for i in range(k)]

    n_components, labels = connected_components(csr_matrix(adj), directed=False, return_labels=True)

    if n_components == 1:
        return [np.arange(k)]

    return [np.where(labels == c)[0] for c in range(n_components)]


@lru_cache(maxsize=128)
def _simulate_component_cached(
    probs_tuple: tuple[float, ...],
    corr_flat: tuple,
    comp_size: int,
    n_simulations: int,
    min_simulations: int,
    max_simulations: int,
    convergence_threshold: float,
) -> float:
    """Run Sobol QMC + Cholesky for a single correlated component."""
    corr_matrix = np.array(corr_flat).reshape(comp_size, comp_size)
    cholesky = _get_cholesky_decomposition(tuple(corr_matrix.flatten()))

    with _CACHE_LOCK:
        if comp_size not in _SOBOL_CACHE:
            _SOBOL_CACHE[comp_size] = qmc.Sobol(d=comp_size, seed=42)
        sobol = _SOBOL_CACHE[comp_size]

    total_samples = min(n_simulations, max_simulations)
    norm_samples = norm.ppf(sobol.random(n=total_samples))

    correlated_samples = norm_samples @ cholesky.T
    prob_array = np.array(probs_tuple)
    admission_outcomes = norm.cdf(correlated_samples) < prob_array[np.newaxis, :]

    return _calculate_converged_probability(
        ~np.any(admission_outcomes, axis=1),
        min_simulations,
        total_samples,
        convergence_threshold,
    )


def run_monte_carlo_simulation(
    selected_schools: list[dict[str, Any]],
    correlation_matrix: pd.DataFrame,
    pair_weight_matrix: pd.DataFrame | None = None,
    n_simulations: int = int(MONTE_CARLO_DEFAULTS["n_simulations"]),
    min_simulations: int = int(MONTE_CARLO_DEFAULTS["min_simulations"]),
    max_simulations: int = int(MONTE_CARLO_DEFAULTS["max_simulations"]),
    convergence_threshold: float = MONTE_CARLO_DEFAULTS["convergence_threshold"],
) -> tuple[float, float]:
    if not selected_schools:
        return 0.0, 0.0

    correlated_schools, independent_schools = _categorize_schools(
        selected_schools, correlation_matrix
    )

    correlated_probs = [round(float(s.get("probability", 0.0)), 6) for s in correlated_schools]
    independent_probs = [round(float(s.get("probability", 0.0)), 6) for s in independent_schools]

    corr_matrix_flat = _get_correlation_matrix(
        correlated_schools, correlation_matrix, pair_weight_matrix
    )

    k = len(correlated_probs)

    return _run_monte_carlo_simulation_cached(
        tuple(correlated_probs),
        tuple(independent_probs),
        corr_matrix_flat,
        k,
        n_simulations,
        min_simulations,
        max_simulations,
        round(float(convergence_threshold), 6),
    )


@lru_cache(maxsize=256)
def _run_monte_carlo_simulation_cached(
    correlated_probs: tuple[float, ...],
    independent_probs: tuple[float, ...],
    corr_matrix_flat: tuple | None,
    k: int,
    n_simulations: int,
    min_simulations: int,
    max_simulations: int,
    convergence_threshold: float,
) -> tuple[float, float]:
    independent_rejection_prob = (
        np.prod(1.0 - np.array(independent_probs)) if independent_probs else 1.0
    )

    if k == 0:
        correlated_rejection_prob = 1.0
    elif k == 1:
        correlated_rejection_prob = 1.0 - correlated_probs[0]
    elif corr_matrix_flat is None:
        correlated_rejection_prob = np.prod(1.0 - np.array(correlated_probs))
    else:
        correlated_rejection_prob = _simulate_correlated_schools(
            correlated_probs,
            corr_matrix_flat,
            k,
            n_simulations,
            min_simulations,
            max_simulations,
            convergence_threshold,
        )

    total_rejection_prob = correlated_rejection_prob * independent_rejection_prob
    return total_rejection_prob, 1.0 - total_rejection_prob


def _simulate_correlated_schools(
    probabilities: tuple[float, ...],
    corr_matrix_flat: tuple,
    k: int,
    n_simulations: int,
    min_simulations: int,
    max_simulations: int,
    convergence_threshold: float,
) -> float:
    corr_matrix = np.array(corr_matrix_flat).reshape(k, k)

    # ── Optimization 2: block-diagonal decomposition ──
    # Split into connected components where |correlation| is meaningful.
    # Each component runs its own low-dim Sobol; results multiply together.
    components = _find_correlation_components(corr_matrix)

    if len(components) == 1 and len(components[0]) == k:
        # No decomposition possible — run full k-dimensional simulation
        return _simulate_component_cached(
            probabilities,
            corr_matrix_flat,
            k,
            n_simulations,
            min_simulations,
            max_simulations,
            convergence_threshold,
        )

    # Run each component independently, multiply rejection probs
    total_rejection = 1.0
    for indices in components:
        comp_probs = tuple(probabilities[i] for i in indices)
        comp_size = len(indices)

        if comp_size == 1:
            total_rejection *= 1.0 - comp_probs[0]
        else:
            comp_corr = corr_matrix[np.ix_(indices, indices)]
            comp_rejection = _simulate_component_cached(
                comp_probs,
                tuple(np.round(comp_corr.flatten(), 6)),
                comp_size,
                n_simulations,
                min_simulations,
                max_simulations,
                convergence_threshold,
            )
            total_rejection *= comp_rejection

    return total_rejection


def _calculate_converged_probability(
    rejection_events: np.ndarray,
    min_simulations: int,
    max_samples: int,
    convergence_threshold: float,
    batch_size: int | None = None,
) -> float:
    if batch_size is None:
        batch_size = int(MONTE_CARLO_DEFAULTS["batch_size"])
    cumulative_rejections = np.cumsum(rejection_events)
    converged_n = min_simulations

    for n in range(min_simulations, max_samples, batch_size):
        if n <= min_simulations:
            continue

        window_size = min(batch_size, n - batch_size)
        prev_prob = cumulative_rejections[n - window_size] / (n - window_size)
        curr_prob = cumulative_rejections[n] / n

        extreme_early = (curr_prob < 0.005 or curr_prob > 0.995) and n >= int(min_simulations * 1.5)

        if (
            abs(curr_prob - prev_prob) < convergence_threshold
            or ((curr_prob < 0.01 or curr_prob > 0.99) and n >= min_simulations * 2)
            or extreme_early
        ):
            converged_n = n
            break

    converged_n = max(1, converged_n)
    return cumulative_rejections[converged_n - 1] / converged_n
