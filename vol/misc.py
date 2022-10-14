from typing import Union, Optional, Iterable, Callable
from numpy import float_
from numpy.typing import NDArray
from dataclasses import dataclass
import math
import numpy as np
from scipy import stats         # type: ignore

FloatArray = NDArray[float_]
Floats = Union[float, FloatArray]


def tkgrid(
    t: Union[float, Iterable[float]],
    k: Union[float, Iterable[float]],
    flat: bool = False
) -> tuple[NDArray[float_], NDArray[float_]]:
    """Returns a grid with coordinates `(t, k)` constructed from the arrays
    `t`, `k` (can be also scalars).

    This is a convenient wrapper over `numpy.meshgrid(indexing='ij')`.

    Example:
        `BlackScholes(s=1, sigma=0.2).call_price(*tkgrid(t=1, k=[0.9, 1]))`
    """
    T, K = np.meshgrid(np.array(t), np.array(k), indexing="ij")
    if flat:
        return T.flatten(), K.flatten()
    return T, K


def iv_to_totalvar(
    s: float,
    t: Union[float, NDArray[float_]],
    k: Union[float, NDArray[float_]],
    iv: Union[float, NDArray[float_]],
    r: float = 0
) -> tuple[Union[float, NDArray[float_]], Union[float, NDArray[float_]]]:
    """Volatility surface in `(log_moneyness, total_variance)` coordinates.

    If the array `iv` represents a volatility surface as a function of
    `(t, k)`, where `t` is expiration time and `k` is strike, then this
    function returns two arrays `x` and `w`, where
        `x = log(k/f_t)` is the log-moneyness, with `f_t = exp(rt)s` being the
            forward price,
        `w = t*iv^2` is the total implied variance.

    This function is useful for, e.g., the SVI model.

    Args:
        s: Current base asset price.
        t: Expiration time.
        k: Strike.
        iv: Implied volatility.
        r: Interest rate.

    Returns:
        A tuple `(x, w)` of log-moneynesses and total implied variances.

    Notes:
        For computation on a grid, use
        `iv_to_totalvar(s, *vol_grid(t, k), iv)`,
        where `t` and `k` are 1-D arrays with grid `iv` is a 2-D of implied
        volatilities.
    """
    x = np.log(k/(s*np.exp(r*t)))
    w = t*iv**2
    return x, w


def totalvar_to_iv(
    s: float,
    t: Union[float, NDArray[float_]],
    x: Union[float, NDArray[float_]],
    w: Union[float, NDArray[float_]],
    r: float = 0
) -> tuple[Union[float, NDArray[float_]], Union[float, NDArray[float_]]]:
    """Converts `(log_moneyness, total_var)` to `(strike, implied_vol)`.

    This function does the change of coordinates reverse to `iv_to_totalvar`.
    See its docstring for the description of the parameters.
    """
    k = np.exp(x+r*t)*s
    iv = np.sqrt(w/t)
    return k, iv


@dataclass
class MCResult:
    """Results of Monte-Carlo simulation.

    Attributes:
        x: Simulation average (what we want to find).
        error: Margin of error, i.e. the confidence interval for the sought-for
            value is `[x-error, x+error]`.
        conf_prob: Confidence probability of the confidence interval.
        success: True if the desired accuracy has been achieved in simulation.
        iterations: Number of random realizations simulated.
        control_coef: Control variate coefficient (if it was used).
    """
    x: float
    error: float
    conf_prob: float
    success: bool
    iterations: int
    control_coef: Optional[float] = None


def vectorize_path_function(
    f: Callable[[NDArray[float_]], float]
) -> Callable[[NDArray[float_]], NDArray[float_]]:
    """Makes a path functional applicable to a batch of paths.

    Args:
        f: A function which is applied to a 1-D array representing a path of
            a random process and returning a scalar value or an array.

    Returns:
        Function `g` which can be applied to an array `S` of shape `(m, n)`,
        which represents `n` paths of a random process each consisting of `m`
        points. The returned value of `g` represents the result of applying `f`
        to each path from `S`.
        If the supplied function `f` is scalar-valued, then `g` returns a 1-D
        array of length `n`. If `f` is vector-valued and returns an array of
        shape `(d1, d2, ...)`, then `g` returns an array of shape
        `(n, d1, d2, ...)`.

    Notes:
        This is useful in the Monte-Carlo method when paths are simulated in
        batches (see `monte_carlo`). However, the resulting function usually
        works slow because it does not make use of NumPy's vectorization.
    """
    def g(S):
        np.stack([f(s) for s in S.T], axis=0)
    return g


# TODO Allow f to be vector-valued
def monte_carlo(simulator: Callable[[int], NDArray[float_]],
                f: Callable[[NDArray[float_]], NDArray[float_]],
                abs_err: float = 1e-3,
                rel_err: float = 1e-3,
                conf_prob: float = 0.95,
                batch_size: int = 10000,
                max_iter: int = 10000000,
                control_f: Optional[
                    Callable[[NDArray[float_]], NDArray[float_]]] = None,
                control_estimation_iter: int = 5000) -> MCResult:
    """The Monte-Carlo method for random processes.

    This function computes the expected value `E(f(X))`, where `f` is the
    provided function and `X` is a random process which simulated by calling
    `simulator`.

    Simulation is performed in batches of random paths to allow speedup by
    vectorization. One batch is obtained in one call of `simulator`. Simulation
    is stopped when the maximum allowed number of path has been exceeded or the
    method has converged in the sense that
        `error < abs_err + x*rel_err`
    where `x` is the current estimated mean value, `error` is the margin of
    error with given confidence probability, i.e. `error = z*s/sqrt(n)`, where
    `z` is the critical value, `s` is the standard error, `n` is the number of
    paths simulated so far.

    It is also possible to provide a control variate, so that the desired value
    will be estimated as
        `E(f(X) - theta*control_f(X))`
    (this helps to reduce the variance). The optimal coefficient `theta` is
    estimated by running a separate Monte-Carlo method with a small number of
    iterations. The random variable corresponding to `control_f` must have zero
    expectation.

    Args:
        simulator: A function which produces random paths. It must accept a
            single argument `n` which is the number of realizations to simulate
            (will be called with `n=batch_size` or `n=control_estimation_iter`)
            and return an array of shape `(n, d)` where `d` is the number of
            sampling points in one path.
        f: Function to apply to the simulated realizations. It must accept an
            a batch of simulated paths (an array of shape `(bath_size, d)`)
            and return an array of size `n`.`.
        abs_err: Desired absolute error.
        rel_err: Desired relative error.
        conf_prob: Desired confidence probability.
        batch_size: Number of random realizations returned in one call to
            `simulator`.
        max_iter: Maximum allowed number of simulated realizations. The desired
            errors may may be not reached if more than `max_iter` paths are
            required.
      control_f: A control variate. Must satisfy the same requirements as `f`.
      control_estimation_iter: Number of random realizations for estimating
        `theta`.

    Returns:
      An MCResult structure with simulation result.
    """
    z: float = stats.norm.ppf((1+conf_prob)/2)    # critical value
    x: float = 0                                  # current mean
    x_sq: float = 0                               # current mean of squares
    s: float = 0                                  # current standard error
    n: int = 0                                    # batches counter
    theta: Optional[float] = None                 # control variate coefficient

    # Estimation of control variate coefficient `theta`
    if control_f is not None:
        S = simulator(control_estimation_iter)
        c = np.cov(f(S), control_f(S))
        theta = c[0, 1] / c[1, 1]

    # Main loop
    while (n == 0 or
           (z*s/math.sqrt(n*batch_size) > abs_err + abs(x)*rel_err
            and n*batch_size < max_iter)):
        S = simulator(batch_size)
        if control_f is not None:
            y = f(S) - theta*control_f(S)
        else:
            y = f(S)
        x = (x*n + np.mean(y))/(n+1)
        x_sq = (x_sq*n + np.mean(y**2))/(n+1)
        s = math.sqrt(x_sq - x**2)
        n += 1

    return MCResult(x=x,
                    error=z*s/math.sqrt(n*batch_size),
                    success=(n*batch_size < max_iter),
                    conf_prob=conf_prob,
                    iterations=n*batch_size,
                    control_coef=theta)
