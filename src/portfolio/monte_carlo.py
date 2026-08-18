import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2, norm, qmc
from scipy.stats import t as t_dist

from src.portfolio.config import (
    MONTE_CARLO_DEFAULTS,
)
from src.utils.numeric import prob_round

_SHRINK_LAMBDA: float = 5.0

_COMPONENT_THRESHOLD: float = 0.03

_T_COPULA_NU: float = 4.0


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
def _simulate_full_with_prestige_cached(
    probs_tuple: tuple[float, ...],
    prestige_tuple: tuple[float, ...],
    corr_flat: tuple,
    k: int,
    n_simulations: int,
    min_simulations: int,
    max_simulations: int,
    convergence_threshold: float,
    t_df: float,
) -> tuple[float, float]:
    corr_matrix = np.array(corr_flat).reshape(k, k)
    cholesky = _get_cholesky_decomposition(tuple(corr_matrix.flatten()))

    total_samples = min(n_simulations, max_simulations)
    sobol = qmc.Sobol(d=k + 1, seed=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*balance properties of Sobol.*")
        raw_samples = sobol.random(n=total_samples)  # (n, k+1)
    prob_array = np.array(probs_tuple)

    if t_df > 0:
        Z = norm.ppf(raw_samples[:, :k]) @ cholesky.T
        W = chi2.ppf(raw_samples[:, k], df=t_df)
        t_samples = Z / np.sqrt(W[:, np.newaxis] / t_df)
        u_samples = t_dist.cdf(t_samples, df=t_df)
        admission_outcomes = u_samples < prob_array[np.newaxis, :]
    else:
        Z = norm.ppf(raw_samples[:, :k]) @ cholesky.T
        admission_outcomes = norm.cdf(Z) < prob_array[np.newaxis, :]

    all_reject = ~np.any(admission_outcomes, axis=1)
    prestige_array = np.array(prestige_tuple)
    best_prestige_per_sample = np.where(
        admission_outcomes,
        prestige_array[np.newaxis, :],
        -1.0,
    ).max(axis=1)
    best_prestige_per_sample = np.maximum(best_prestige_per_sample, 0.0)

    p_reject, ev_prestige, _converged_n = _calculate_converged_estimates(
        all_reject,
        best_prestige_per_sample,
        min_simulations,
        total_samples,
        convergence_threshold,
    )

    return p_reject, ev_prestige


@lru_cache(maxsize=128)
def _simulate_component_cached(
    probs_tuple: tuple[float, ...],
    corr_flat: tuple,
    comp_size: int,
    n_simulations: int,
    min_simulations: int,
    max_simulations: int,
    convergence_threshold: float,
    t_df: float,
) -> float:
    corr_matrix = np.array(corr_flat).reshape(comp_size, comp_size)
    cholesky = _get_cholesky_decomposition(tuple(corr_matrix.flatten()))

    total_samples = min(n_simulations, max_simulations)
    sobol_dim = comp_size + (1 if t_df > 0 else 0)
    sobol = qmc.Sobol(d=sobol_dim, seed=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*balance properties of Sobol.*")
        raw_samples = sobol.random(n=total_samples)

    Z = norm.ppf(raw_samples[:, :comp_size]) @ cholesky.T
    if t_df > 0:
        W = chi2.ppf(raw_samples[:, comp_size], df=t_df)
        t_samples = Z / np.sqrt(W[:, np.newaxis] / t_df)
        u_samples = t_dist.cdf(t_samples, df=t_df)
    else:
        u_samples = norm.cdf(Z)

    prob_array = np.array(probs_tuple)
    admission_outcomes = u_samples < prob_array[np.newaxis, :]

    prob, _n = _calculate_converged_probability(
        ~np.any(admission_outcomes, axis=1),
        min_simulations,
        total_samples,
        convergence_threshold,
    )
    return prob


def run_monte_carlo_simulation(
    selected_schools: list[dict[str, Any]],
    correlation_matrix: pd.DataFrame,
    pair_weight_matrix: pd.DataFrame | None = None,
    n_simulations: int = int(MONTE_CARLO_DEFAULTS["n_simulations"]),
    min_simulations: int = int(MONTE_CARLO_DEFAULTS["min_simulations"]),
    max_simulations: int = int(MONTE_CARLO_DEFAULTS["max_simulations"]),
    convergence_threshold: float = MONTE_CARLO_DEFAULTS["convergence_threshold"],
    school_prestige_scores: list[float] | None = None,
    compute_ev: bool = True,
) -> tuple[float, float, float | None]:
    if not selected_schools:
        return (0.0, 0.0, 0.0 if school_prestige_scores is not None else None)

    if (
        school_prestige_scores is None
        and correlation_matrix is not None
        and not correlation_matrix.empty
    ):
        from src.portfolio.prestige import calculate_prestige_score

        school_prestige_scores = [calculate_prestige_score([s]) for s in selected_schools]

    correlated_schools, independent_schools = _categorize_schools(
        selected_schools, correlation_matrix
    )

    correlated_probs = [prob_round(s.get("probability"), ndigits=6) for s in correlated_schools]
    independent_probs = [prob_round(s.get("probability"), ndigits=6) for s in independent_schools]

    if school_prestige_scores is not None and len(school_prestige_scores) == len(selected_schools):
        prestige_by_uni = {
            s.get("university", ""): p
            for s, p in zip(selected_schools, school_prestige_scores, strict=True)
        }
        all_schools = correlated_schools + independent_schools
        all_probs = [prob_round(s.get("probability"), ndigits=6) for s in all_schools]
        all_prestige = [prestige_by_uni[s.get("university", "")] for s in all_schools]
        k_all = len(all_probs)

        if k_all == 0:
            return (0.0, 0.0, 0.0)

        if k_all == 1:
            p = all_probs[0]
            ev = all_prestige[0] * p if compute_ev else None
            return (1.0 - p, p, ev)

        full_corr = np.eye(k_all, dtype=float)
        k_corr = len(correlated_probs)
        if k_corr > 1:
            corr_keys = [f"{s['university']} - {s['major']}" for s in correlated_schools]
            sub_corr = correlation_matrix.loc[corr_keys, corr_keys].values.astype(float)
            if pair_weight_matrix is not None and not pair_weight_matrix.empty:
                try:
                    weight_sub = pair_weight_matrix.loc[corr_keys, corr_keys]
                    n_ij = weight_sub.values.astype(float)
                    shrinkage = n_ij / (n_ij + _SHRINK_LAMBDA)
                    sub_corr = sub_corr * shrinkage
                except KeyError:
                    pass
            np.fill_diagonal(sub_corr, 1.0)
            full_corr[:k_corr, :k_corr] = sub_corr

        p_reject, ev_prestige = _simulate_full_with_prestige_cached(
            tuple(all_probs),
            tuple(all_prestige),
            tuple(np.round(full_corr.flatten(), 6)),
            k_all,
            n_simulations,
            min_simulations,
            max_simulations,
            round(float(convergence_threshold), 6),
            _T_COPULA_NU,
        )
        return (p_reject, 1.0 - p_reject, ev_prestige if compute_ev else None)

    corr_matrix_flat = _get_correlation_matrix(
        correlated_schools, correlation_matrix, pair_weight_matrix
    )
    k = len(correlated_probs)

    result = _run_monte_carlo_simulation_cached(
        tuple(correlated_probs),
        tuple(independent_probs),
        corr_matrix_flat,
        k,
        n_simulations,
        min_simulations,
        max_simulations,
        round(float(convergence_threshold), 6),
        _T_COPULA_NU,
    )
    return (result[0], result[1], None)


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
    t_df: float,
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
            t_df,
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
    t_df: float,
) -> float:
    corr_matrix = np.array(corr_matrix_flat).reshape(k, k)

    components = _find_correlation_components(corr_matrix)

    if len(components) == 1 and len(components[0]) == k:
        return _simulate_component_cached(
            probabilities,
            corr_matrix_flat,
            k,
            n_simulations,
            min_simulations,
            max_simulations,
            convergence_threshold,
            t_df,
        )

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
                t_df,
            )
            total_rejection *= comp_rejection

    return total_rejection


def _calculate_converged_probability(
    rejection_events: np.ndarray,
    min_simulations: int,
    max_samples: int,
    convergence_threshold: float,
    batch_size: int | None = None,
) -> tuple[float, int]:
    if batch_size is None:
        batch_size = int(MONTE_CARLO_DEFAULTS["batch_size"])
    cumulative_rejections = np.cumsum(rejection_events)
    converged_n = max_samples

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
    return cumulative_rejections[converged_n - 1] / converged_n, converged_n


def _calculate_converged_estimates(
    rejection_events: np.ndarray,
    values: np.ndarray,
    min_simulations: int,
    max_samples: int,
    convergence_threshold: float,
    batch_size: int | None = None,
) -> tuple[float, float, int]:
    if batch_size is None:
        batch_size = int(MONTE_CARLO_DEFAULTS["batch_size"])

    cumulative_rejections = np.cumsum(rejection_events)
    cumulative_values = np.cumsum(values)
    converged_n = max_samples

    for n in range(min_simulations + batch_size, max_samples, batch_size):
        window_size = min(batch_size, n - batch_size)
        prev_reject = cumulative_rejections[n - window_size] / (n - window_size)
        curr_reject = cumulative_rejections[n] / n
        prev_ev = cumulative_values[n - window_size] / (n - window_size)
        curr_ev = cumulative_values[n] / n

        extreme_early = (curr_reject < 0.005 or curr_reject > 0.995) and n >= int(
            min_simulations * 1.5
        )
        reject_ok = (
            abs(curr_reject - prev_reject) < convergence_threshold
            or ((curr_reject < 0.01 or curr_reject > 0.99) and n >= min_simulations * 2)
            or extreme_early
        )
        ev_ok = abs(curr_ev - prev_ev) < convergence_threshold
        if reject_ok and ev_ok:
            converged_n = n
            break

    converged_n = max(1, converged_n)
    p_reject = cumulative_rejections[converged_n - 1] / converged_n
    mean_value = cumulative_values[converged_n - 1] / converged_n
    return float(p_reject), float(mean_value), int(converged_n)
