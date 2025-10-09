from functools import lru_cache
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from src.pages.prediction.school_combination_optimizer_algorithm.optimizer_config import (
    MONTE_CARLO_DEFAULTS,
)

_CHOLESKY_CACHE: dict[tuple[str, ...], Any] = {}
_SOBOL_CACHE: dict[int, qmc.Sobol] = {}


@lru_cache(maxsize=128)
def _get_cholesky_decomposition(matrix_tuple):
    matrix = np.array(matrix_tuple).reshape(int(np.sqrt(len(matrix_tuple))), -1)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        min_legal_eigenvalue = 1e-6
        eigenvalues[eigenvalues <= min_legal_eigenvalue] = min_legal_eigenvalue
        fixed_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(fixed_matrix)))
        fixed_matrix_normalized = diag_inv_sqrt @ fixed_matrix @ diag_inv_sqrt
        return np.linalg.cholesky(fixed_matrix_normalized)


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

    correlated_schools: list[dict[str, Any]] = []
    independent_schools: list[dict[str, Any]] = []

    if correlation_matrix is None or correlation_matrix.empty:
        independent_schools = list(selected_schools)
    else:
        matrix_keys = correlation_matrix.index
        for school in selected_schools:
            key = f"{school['university']} - {school['major']}"
            if key in matrix_keys:
                correlated_schools.append(school)
            else:
                independent_schools.append(school)

    correlated_probs = tuple(round(float(s.get("probability", 0.0)), 6) for s in correlated_schools)
    independent_probs = tuple(
        round(float(s.get("probability", 0.0)), 6) for s in independent_schools
    )

    corr_matrix_flat: Tuple[float, ...] | None = None
    k = len(correlated_probs)

    if k >= 2 and correlation_matrix is not None and not correlation_matrix.empty:
        try:
            school_keys = [f"{s['university']} - {s['major']}" for s in correlated_schools]
            sub_corr_df = correlation_matrix.loc[school_keys, school_keys]
            sub_corr_matrix = sub_corr_df.values.astype(float)

            if (
                pair_weight_matrix is not None
                and set(school_keys).issubset(set(pair_weight_matrix.index))
                and set(school_keys).issubset(set(pair_weight_matrix.columns))
            ):
                try:
                    weights_df = pair_weight_matrix.loc[school_keys, school_keys]
                    sub_corr_matrix = sub_corr_matrix * weights_df.values.astype(float)
                except Exception:
                    pass

            np.fill_diagonal(sub_corr_matrix, 1.0)
            corr_matrix_flat = tuple(np.round(sub_corr_matrix.flatten(), 6))
        except Exception:
            corr_matrix_flat = None

    total_rejection_prob, total_admission_prob = _run_monte_carlo_simulation_cached(
        correlated_probs,
        independent_probs,
        corr_matrix_flat,
        k,
        n_simulations,
        min_simulations,
        max_simulations,
        round(float(convergence_threshold), 6),
    )

    return total_rejection_prob, total_admission_prob


@lru_cache(maxsize=256)
def _run_monte_carlo_simulation_cached(
    correlated_probs: tuple[float, ...],
    independent_probs: tuple[float, ...],
    corr_matrix_flat: Tuple[float, ...] | None,
    k: int,
    n_simulations: int,
    min_simulations: int,
    max_simulations: int,
    convergence_threshold: float,
) -> tuple[float, float]:
    independent_rejection_prob = 1.0
    for prob in independent_probs:
        independent_rejection_prob *= 1.0 - prob

    correlated_rejection_prob = 1.0

    if k >= 2 and corr_matrix_flat is not None:
        try:
            sub_corr_matrix = np.array(corr_matrix_flat, dtype=float).reshape(k, k)
            matrix_tuple = tuple(sub_corr_matrix.flatten())
            cholesky_decomp = _get_cholesky_decomposition(matrix_tuple)

            sobol = _SOBOL_CACHE.get(k)
            if sobol is None:
                sobol = qmc.Sobol(d=k, seed=42)
                _SOBOL_CACHE[k] = sobol

            total_samples = min(int(n_simulations), int(max_simulations))
            norm_samples = norm.ppf(sobol.random(n=total_samples))

            probabilities = np.array(correlated_probs, dtype=float)
            correlated_samples = norm_samples @ cholesky_decomp.T
            admission_outcomes = norm.cdf(correlated_samples) < probabilities[np.newaxis, :]

            all_rejected = ~np.any(admission_outcomes, axis=1)
            cumulative_rejection_count = np.cumsum(all_rejected)

            batch_size = int(MONTE_CARLO_DEFAULTS.get("batch_size", 500))
            converged_n = int(min_simulations)

            for n in range(int(min_simulations), total_samples, batch_size):
                if n > min_simulations:
                    window_size = min(batch_size, n - batch_size)
                    prev_prob = cumulative_rejection_count[n - window_size] / (n - window_size)
                    curr_prob = cumulative_rejection_count[n] / n

                    if abs(curr_prob - prev_prob) < convergence_threshold:
                        converged_n = n
                        break

                    if (curr_prob < 0.01 or curr_prob > 0.99) and n >= int(min_simulations) * 2:
                        converged_n = n
                        break

            correlated_rejection_prob = cumulative_rejection_count[converged_n - 1] / converged_n
        except Exception:
            correlated_rejection_prob = 1.0
            for prob in correlated_probs:
                correlated_rejection_prob *= 1.0 - prob

    elif k == 1:
        correlated_rejection_prob = 1.0 - correlated_probs[0]

    total_rejection_prob = correlated_rejection_prob * independent_rejection_prob
    total_admission_prob = 1.0 - total_rejection_prob

    return total_rejection_prob, total_admission_prob
