from typing import Any, Callable, Optional

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.result import Result
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    DEFAULT_REFERENCE_DIRECTIONS_COUNT,
    PlanConfig,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.context import (
    OptimizationContext,
)
from src.pages.prediction.school_combination_optimizer_algorithm.problem import (
    SchoolSelectionProblem,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    get_cached_reference_directions,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def create_problem(
    schools_data: list[dict[str, Any]],
    plan_config: PlanConfig,
    context: OptimizationContext,
) -> SchoolSelectionProblem:
    return SchoolSelectionProblem(
        all_schools_data=schools_data,
        background_major=context.background_major,
        background_faculty=context.background_faculty,
        max_schools=plan_config.max_schools,
        adaptive_thresholds=context.adaptive_thresholds,
        school_level=context.school_level,
        gpa=context.gpa,
        min_schools=plan_config.min_schools,
    )


def compute_algo_params(
    problem_size: int, population_size: int, n_generations: int
) -> tuple[int, int, Any]:
    n_ref = DEFAULT_REFERENCE_DIRECTIONS_COUNT
    base_pop = population_size
    base_gen = n_generations

    if problem_size < 30:
        pop = max(base_pop, problem_size * 2)
        n_gen = base_gen
    elif problem_size < 50:
        pop = max(base_pop, int(base_pop * 1.2))
        n_gen = int(base_gen * 1.2)
    else:
        pop = base_pop
        n_gen = base_gen

    ref = get_cached_reference_directions("energy", n_dim=5, n_points=n_ref)
    logger.info(
        f"优化参数: problem_size={problem_size}, population_size={pop}, n_generations={n_gen}"
    )
    return pop, n_gen, ref


def run_optimization(
    problem: SchoolSelectionProblem,
    population_size: int,
    n_generations: int,
    safe_execute: Callable[[], Any],
) -> Optional[Result]:
    dynamic_pop_size, dynamic_n_gen, ref_dirs = compute_algo_params(
        len(problem.all_schools_data), population_size, n_generations
    )

    algorithm = NSGA3(
        pop_size=dynamic_pop_size,
        ref_dirs=ref_dirs,
        sampling=BinaryRandomSampling(),
        crossover=HUX(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )

    return safe_execute(
        lambda: minimize(problem, algorithm, ("n_gen", dynamic_n_gen), seed=1, verbose=False),
        error_message="优化过程中出现错误",
    )

