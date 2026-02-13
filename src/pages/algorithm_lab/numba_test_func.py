import numpy as np
from numba import float64, guvectorize, njit, prange, vectorize


@njit(fastmath=False)
def fastmath_off(a):
    res = 0.0
    for i in range(len(a)):
        res += np.exp(a[i])
    return res


@njit(fastmath=True)
def fastmath_on(a):
    res = 0.0
    for i in range(len(a)):
        res += np.exp(a[i])
    return res


@njit
def sum_slice(arr):
    h, w = arr.shape
    res = np.zeros(w)
    for j in range(w):
        tmp = 0.0
        for i in range(h):
            tmp += arr[i, j]
        res[j] = tmp
    return res


@njit(float64(float64[:]))
def explicit_sig(a):
    return np.sum(a)


@njit
def auto_sig(a):
    return np.sum(a)


@guvectorize([(float64[:, :], float64[:, :], float64[:, :])], "(m,n),(n,p)->(m,p)")
def numba_matmul_gu(a, b, res):
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            tmp = 0.0
            for k in range(a.shape[1]):
                tmp += a[i, k] * b[k, j]
            res[i, j] = tmp


@njit
def binary_search_numba(arr, item):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return -1


def binary_search_python(arr, item):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return -1


def python_pi(n):
    acc = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / n


def python_pi_from_arrays(x, y):
    acc = 0
    n = len(x)
    for i in range(n):
        if (x[i] * x[i] + y[i] * y[i]) < 1.0:
            acc += 1
    return 4.0 * acc / n


@njit
def numba_pi(n):
    acc = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / n


@njit
def count_inside_circle(x, y):
    acc = 0
    for i in range(x.shape[0]):
        if (x[i] * x[i] + y[i] * y[i]) < 1.0:
            acc += 1
    return acc


@njit(parallel=True)
def numba_pi_parallel(n):
    acc = 0
    for i in prange(n):
        x = np.random.random()
        y = np.random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / n


@njit(parallel=True)
def count_inside_circle_parallel(x, y):
    acc = 0
    for i in prange(x.shape[0]):
        if (x[i] * x[i] + y[i] * y[i]) < 1.0:
            acc += 1
    return acc


def mandelbrot_python(h, w, max_iter):
    y, x = np.ogrid[-1.4 : 1.4 : h * 1j, -2 : 0.8 : w * 1j]
    c = x + y * 1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)
    for i in range(max_iter):
        z = z**2 + c
        diverge = z * np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
    return divtime


@njit
def mandelbrot_numba(h, w, max_iter):
    image = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            c = complex(-2.0 + j * 2.8 / w, -1.4 + i * 2.8 / h)
            z = 0.0j
            for k in range(max_iter):
                z = z * z + c
                if (z.real * z.real + z.imag * z.imag) >= 4.0:
                    image[i, j] = k
                    break
            else:
                image[i, j] = max_iter
    return image


@njit(parallel=True)
def mandelbrot_numba_parallel(h, w, max_iter):
    image = np.zeros((h, w), dtype=np.int32)
    for i in prange(h):
        for j in range(w):
            c = complex(-2.0 + j * 2.8 / w, -1.4 + i * 2.8 / h)
            z = 0.0j
            for k in range(max_iter):
                z = z * z + c
                if (z.real * z.real + z.imag * z.imag) >= 4.0:
                    image[i, j] = k
                    break
            else:
                image[i, j] = max_iter
    return image


@vectorize([float64(float64, float64)], target="cpu")
def numba_vectorize_add(a, b):
    return a + b


@vectorize([float64(float64, float64)], target="parallel")
def numba_vectorize_add_parallel(a, b):
    return a + b


@njit(parallel=True)
def sobel_numba_parallel(image):
    h, w = image.shape
    out = np.zeros_like(image)
    for i in prange(1, h - 1):
        for j in range(1, w - 1):
            gx = (image[i - 1, j + 1] + 2 * image[i, j + 1] + image[i + 1, j + 1]) - (
                image[i - 1, j - 1] + 2 * image[i, j - 1] + image[i + 1, j - 1]
            )
            gy = (image[i + 1, j - 1] + 2 * image[i + 1, j] + image[i + 1, j + 1]) - (
                image[i - 1, j - 1] + 2 * image[i - 1, j] + image[i - 1, j + 1]
            )
            out[i, j] = np.sqrt(gx**2 + gy**2)
    return out


def sobel_numpy(image):
    out = np.zeros_like(image)
    gx = (image[:-2, 2:] + 2 * image[1:-1, 2:] + image[2:, 2:]) - (
        image[:-2, :-2] + 2 * image[1:-1, :-2] + image[2:, :-2]
    )
    gy = (image[2:, :-2] + 2 * image[2:, 1:-1] + image[2:, 2:]) - (
        image[:-2, :-2] + 2 * image[:-2, 1:-1] + image[:-2, 2:]
    )
    out[1:-1, 1:-1] = np.sqrt(gx * gx + gy * gy)
    return out
