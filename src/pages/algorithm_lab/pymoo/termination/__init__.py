def get_termination(name, *args, **kwargs):
    from src.pages.algorithm_lab.pymoo.termination.default import (
        DefaultMultiObjectiveTermination,
        DefaultSingleObjectiveTermination,
    )
    from src.pages.algorithm_lab.pymoo.termination.fmin import MinimumFunctionValueTermination
    from src.pages.algorithm_lab.pymoo.termination.max_eval import MaximumFunctionCallTermination
    from src.pages.algorithm_lab.pymoo.termination.max_gen import MaximumGenerationTermination
    from src.pages.algorithm_lab.pymoo.termination.max_time import TimeBasedTermination

    TERMINATION = {
        "n_eval": MaximumFunctionCallTermination,
        "n_evals": MaximumFunctionCallTermination,
        "n_gen": MaximumGenerationTermination,
        "n_iter": MaximumGenerationTermination,
        "fmin": MinimumFunctionValueTermination,
        "time": TimeBasedTermination,
        "soo": DefaultSingleObjectiveTermination,
        "moo": DefaultMultiObjectiveTermination,
    }

    if name not in TERMINATION:
        raise Exception("Termination not found.")

    return TERMINATION[name](*args, **kwargs)
