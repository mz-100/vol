from typing import Union, Optional
from numpy import float_
from numpy.typing import NDArray
from dataclasses import dataclass
import math
import numpy as np
import scipy.stats as st    # type: ignore
from scipy import optimize  # type: ignore


@dataclass
class BlackScholes:
    """The Black-Scholes model.

    The base asset is stock which under the pricing measure follows the SDE
        `d(S_t) = r*S_t*dt + sigma*S_t*d(W_t)`
    where `r` is the interest rate, `sigma>0` is the volatility.

    Attributes:
        s : Initial price, i.e. S_0.
        sigma : Volatility.
        r : Risk-free interest rate.

    Methods:
        call_price: Computes the price of a call option.
        call_delta: Computes delta of a call option.
        call_theta: Computes theta of a call option.
        call_vega: Computes vega of a call option.
        call_gamma: Computes gamma of a call option.
        call_iv: Computes the implied volatility of a call option.
        simulate: Simulates paths.
    """
    s: float
    sigma: float
    r: float = 0

    def _d1(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """Computes `d_1` from the Black-Scholes formula."""
        return ((np.log(self.s/k) + (self.r + 0.5*self.sigma**2)*t) /
                (self.sigma*np.sqrt(t)))

    def _d2(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """Computes `d_2` from the Black-Scholes formula."""
        return self._d1(t, k) - self.sigma*np.sqrt(t)

    def call_price(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """Computes the price of a call option.

        Args:
            t: Expiration time (float or ndarray).
            k: Strike (float or ndarray).

        Returns:
            Call option price. If `t` and/or `k` are arrays, applies NumPy
            broadcasting rules and returns the array of prices.

        Notes:
            For computation on a grid, use `call_price(*vol_grid(t, k))`, where
            `t` and `k` are 1-D arrays with the grid coordinates.
        """
        return (self.s*st.norm.cdf(self._d1(t, k)) -
                np.exp(-self.r*t)*k*st.norm.cdf(self._d2(t, k)))

    def call_delta(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """Computes delta of a call options.

        See `call_price` for the description of arguments and return value.
        """
        return st.norm.cdf(self._d1(t, k))

    def call_theta(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """Computes theta of a call options.

        See `call_price` for the description of arguments and return value.
        """
        return (-self.s*st.norm.pdf(self._d1(t, k)) *
                self.sigma/(2*np.sqrt(t)) -
                self.r*np.exp(-self.r*t)*k*st.norm.cdf(self._d2(t, k)))

    def call_vega(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """Computes vega of a call options.

        See `call_price` for the description of arguments and return value.
        """
        return self.s * st.norm.pdf(self._d1(t, k)) * np.sqrt(t)

    def call_gamma(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """Computes gamma of a call options.

        See `call_price` for the description of arguments and return value.
        """
        return st.norm.pdf(self._d1(t, k)) / (self.s*self.sigma*np.sqrt(t))

    def call_iv(
        self,
        c: Union[float, NDArray[float_]],
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]],
        iv_approx_bounds: Optional[tuple[float, float]] = None
    ) -> Union[float, NDArray[float_]]:
        """Computes the implied volatility of a call option.

        This function wraps over the module's `call_iv` function, using the
        initial price and risk-free interest rate from the class attributes,
        and ignoring the class attribute `sigma` (volatility).

        See `call_iv` for the description of parameters and return value.
        """
        return call_iv(self.s, self.r, c, t, k, iv_approx_bounds)

    def simulate(
        self,
        t: float,
        steps: int,
        paths: int
    ) -> NDArray[float_]:
        """Simulates paths of the price process.

        Args:
            t: Time horizon.
            steps: Number of simulation points minus 1, i.e. paths are sampled
                at `t_i = i*dt`, where `i = 0, ..., steps`, `dt = t/steps`.
            paths: Number of paths to simulate.

        Returns:
            An array `s` of shape `(steps+1, paths)`, where `s[i, j]` is the
            value of the j-th path at point `t_i`.
        """
        dt = t/steps
        # Increments of the log-price
        z = ((self.r-0.5*self.sigma**2)*dt +
             np.random.standard_normal(size=(steps, paths)) *
             self.sigma*math.sqrt(dt))
        bm = np.concatenate([np.zeros((1, paths)), np.cumsum(z, axis=0)])
        return self.s*np.exp(bm)


"""
The following functions are used for computation of implied volatility.
* call_price implements the Black-Scholes formula.
* call_iv returns the implied volatility.
* _call_iv_approx gives an initial estimate of IV.
* _call_iv_f is the objective function for root finding
    (Black-Scholes price - market price).
* _call_iv_fprime is the derivative of the objective function.
"""


def call_price(
    s: Union[float, NDArray[float_]],
    sigma: Union[float, NDArray[float_]],
    t: Union[float, NDArray[float_]],
    k: Union[float, NDArray[float_]],
    r: Union[float, NDArray[float_]] = 0
) -> Union[float, NDArray[float_]]:
    """Computes the Black--Scholes price of a call option.

    Args:
        s: Underlying asset price.
        sigma: Volatility.
        t: Expiration time.
        k: Strike.
        r: Interest rate.

    Returns:
        Call option price. If the arguments are arrays, applies NumPy
        broadcasting rules and returns the array of prices.

    Notes:
        This functions does the same computation as BlackScholes.call_price,
        but it allows to vectorize `s`, `sigma` and `r`.
    """
    d1 = (np.log(s/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    return s*st.norm.cdf(d1) - np.exp(-r*t)*k*st.norm.cdf(d2)


# For compatibility
_call_price = call_price


def _call_iv_approx(
    s: Union[float, NDArray[float_]],
    r: Union[float, NDArray[float_]],
    c: Union[float, NDArray[float_]],
    t: Union[float, NDArray[float_]],
    k: Optional[Union[float, NDArray[float_]]] = None
) -> Union[float, NDArray[float_]]:
    """Computes an approximation of the implied volatility of a call option.

    This is an auxiliary function to find an initial estimate of the implied
    volatility to be used in Newton's method in function `call_iv`.

    See `call_iv` for description of arguments and return value.

    Notes:
        If `k` is `None`, the Brenner-Subrahmanyam formula is used. If ` k` is
        not `None`, the Corrado-Miller formula is used.
    """
    if k is not None:
        z = np.exp(-r*t)*k
        a = np.maximum(0, (c - (s-z)/2)**2 - (s-z)**2/np.pi)
        return (np.sqrt(2*np.pi/t)/(s + z) * (c - (s-z)/2 + np.sqrt(a)))
    else:
        return 2.5*c/(s*np.sqrt(t))


def _call_iv_f(
    sigma: Union[float, NDArray[float_]],
    s: Union[float, NDArray[float_]],
    r: Union[float, NDArray[float_]],
    c: Union[float, NDArray[float_]],
    t: Union[float, NDArray[float_]],
    k: Union[float, NDArray[float_]]
) -> Union[float, NDArray[float_]]:
    return call_price(s, sigma, t, k, r) - c


def _call_iv_fprime(
    sigma: Union[float, NDArray[float_]],
    s: Union[float, NDArray[float_]],
    r: Union[float, NDArray[float_]],
    c: Union[float, NDArray[float_]],
    t: Union[float, NDArray[float_]],
    k: Union[float, NDArray[float_]]
) -> Union[float, NDArray[float_]]:
    d1 = (np.log(s/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    return s*st.norm.pdf(d1)*np.sqrt(t)


def call_iv(
    s: float,
    r: float,
    c: Union[float, NDArray[float_]],
    t: Union[float, NDArray[float_]],
    k: Union[float, NDArray[float_]],
    iv_approx_bounds: Optional[tuple[float, float]] = None
) -> Union[float, NDArray[float_]]:
    """Computes the implied volatility of a call option.

    This function uses Newton's root finding method to invert the Black-Scholes
    formula with respect to `sigma`.

    Args:
        s: Underlying asset price.
        r: Risk-free interest rate.
        c: Market call option price.
        t: Expiration time.
        k: Strike (optional).
        iv_approx_bounds: A tuple `(min, max)` or `None` such that the initial
            guess of the implied volatility will be truncated if it is outside
            the interval `[min, max]`. This is useful for extreme strikes or
            maturities, when the approximate formula gives unrealistic results.
            If None, no truncation will be applied.

    Returns:
        The implied volatility if Newton's method converged successfully;
        otherwise returns NaN. If the arguments are arrays, which must be of
        the same shape, an array of the same shape is returned, where each
        element is the implied volatility or NaN.

    Notes:
        For computation on a grid of option prices, use
        `call_iv(s, r, c,  *vol_grid(t, k))`, where `t` and `k` are 1-D arrays
        with grid coordinates, and `c` is a 2-D of option prices.
    """
    iva = _call_iv_approx(s, r, c, t, k)
    if iv_approx_bounds is None:
        x0 = iva
    else:
        x0 = np.minimum(np.maximum(iva, iv_approx_bounds[0]),
                        iv_approx_bounds[1])
    res = optimize.newton(
        func=_call_iv_f,
        args=(s, r, c, t, k),
        x0=x0,
        fprime=_call_iv_fprime,
        full_output=True)

    if hasattr(res, "root"):  # vector-valued arguments were supplied
        return np.where(res.converged, res.root, np.NaN)
    else:
        return res[0] if res[1].converged else np.NaN
