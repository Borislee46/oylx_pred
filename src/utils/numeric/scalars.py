import math

from numba import jit

PROB_EPS = 1e-6
FLOAT_EQ_RTOL = 1e-9
FLOAT_EQ_ATOL = 1e-12
CLOSE_TO_INT_TOL = 1e-9


@jit(nopython=True, cache=True)
def clip_probability(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


@jit(nopython=True, cache=True)
def clip_scalar(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@jit(nopython=True, cache=True)
def logit(p: float) -> float:
    if p < PROB_EPS:
        p = PROB_EPS
    elif p > 1.0 - PROB_EPS:
        p = 1.0 - PROB_EPS
    return math.log(p / (1.0 - p))


@jit(nopython=True, cache=True)
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@jit(nopython=True, cache=True)
def sigmoid_k(x: float, k: float, x0: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


@jit(nopython=True, cache=True)
def min_max_scale(x: float, x_min: float, x_max: float) -> float:
    if x_max <= x_min:
        return 0.5
    return (x - x_min) / (x_max - x_min)


@jit(nopython=True, cache=True)
def float_eq(a: float, b: float, rtol: float = FLOAT_EQ_RTOL, atol: float = FLOAT_EQ_ATOL) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


@jit(nopython=True, cache=True)
def is_close_to_int(x: float, tol: float = CLOSE_TO_INT_TOL) -> bool:
    return abs(x - round(x)) <= tol


@jit(nopython=True, cache=True)
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0.0:
        return default
    return a / b
