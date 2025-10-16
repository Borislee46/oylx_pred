from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    MONTE_CARLO_DEFAULTS,
)

_CHOLESKY_CACHE: dict[tuple, Any] = {}
_SOBOL_CACHE: dict[int, qmc.Sobol] = {}


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
    correlated, independent = [], []

    for school in schools:
        key = f"{school['university']} - {school['major']}"
        if key in matrix_keys:
            correlated.append(school)
        else:
            independent.append(school)

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

    if (
        pair_weight_matrix is not None
        and set(school_keys).issubset(pair_weight_matrix.index)
        and set(school_keys).issubset(pair_weight_matrix.columns)
    ):
        weights = pair_weight_matrix.loc[school_keys, school_keys].values.astype(float)
        corr_matrix *= weights

    np.fill_diagonal(corr_matrix, 1.0)
    return tuple(np.round(corr_matrix.flatten(), 6))


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

    correlated_probs = tuple(round(float(s.get("probability", 0.0)), 6) for s in correlated_schools)
    independent_probs = tuple(
        round(float(s.get("probability", 0.0)), 6) for s in independent_schools
    )

    corr_matrix_flat = _get_correlation_matrix(
        correlated_schools, correlation_matrix, pair_weight_matrix
    )

    k = len(correlated_probs)

    return _run_monte_carlo_simulation_cached(
        correlated_probs,
        independent_probs,
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
    cholesky = _get_cholesky_decomposition(tuple(corr_matrix.flatten()))

    sobol = _SOBOL_CACHE.get(k, qmc.Sobol(d=k, seed=42))
    _SOBOL_CACHE[k] = sobol

    total_samples = min(n_simulations, max_simulations)
    norm_samples = norm.ppf(sobol.random(n=total_samples))

    correlated_samples = norm_samples @ cholesky.T
    prob_array = np.array(probabilities)
    admission_outcomes = norm.cdf(correlated_samples) < prob_array[np.newaxis, :]

    return _calculate_converged_probability(
        ~np.any(admission_outcomes, axis=1),
        min_simulations,
        total_samples,
        convergence_threshold,
    )


def _calculate_converged_probability(
    rejection_events: np.ndarray,
    min_simulations: int,
    max_samples: int,
    convergence_threshold: float,
    batch_size: int = 500,
) -> float:
    cumulative_rejections = np.cumsum(rejection_events)
    converged_n = min_simulations

    for n in range(min_simulations, max_samples, batch_size):
        if n <= min_simulations:
            continue

        window_size = min(batch_size, n - batch_size)
        prev_prob = cumulative_rejections[n - window_size] / (n - window_size)
        curr_prob = cumulative_rejections[n] / n

        if (
            abs(curr_prob - prev_prob) < convergence_threshold
            or (curr_prob < 0.01 or curr_prob > 0.99)
            and n >= min_simulations * 2
        ):
            converged_n = n
            break

    return cumulative_rejections[converged_n - 1] / converged_n
