from typing import Union
from numpy import float_
from numpy.typing import NDArray
from dataclasses import dataclass
import numpy as np
import scipy.stats as st  # type: ignore
from scipy import optimize, special  # type: ignore
from . import blackscholes


@dataclass
class CEV:
    """The CEV (Constant Elasticity of Variance) model.

    The base asset is stock which under the pricing measure follows the SDE
        `d(S_t) = r*S_t*dt + sigma*S_t^beta*d(W_t)`
    where `r` is the interest rate, `sigma>0` and `beta>=0` are parameters.

    Attributes:
        s: Initial price, i.e. S_0.
        sigma: Volatility.
        beta: Parameter which controls the skew.
        r: Risk-free interest rate.

    Methods:
        vanish_probability: Computes the probability to cross zero.
        call_price: Computes the price of a call option.
        iv: Computes the implied volatility produced by the model.
        calibrate: Calibrates parameters of the model.
        simulate: Simulates paths by sampling from the exact distribution.
        monte_carlo: Computes expectation by the Monte-Carlo method.
    """
    s: float
    sigma: float
    beta: float
    r: float = 0

    def _remove_drift(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> tuple[Union[float, NDArray[float_]], Union[float, NDArray[float_]]]:
        """Returns adjusted time to expiration and strike which reduce
        computations to the driftless model.

        Notes:
            If `t`, `k` are arrays, usual broadcasting rules are applied.
        """
        if self.r == 0:
            return t, k
        t_adj = (np.exp(2*self.r*(self.beta-1)*t)-1)/(2*self.r*(self.beta-1))
        k_adj = np.exp(-self.r*t)*k
        return t_adj, k_adj

    def vanish_probability(
        self,
        t: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """Computes the probability of reaching zero by time `t`.

        Args:
            t: Time (float or ndarray).

        Returns:
            Probability of reaching zero by `t`. If `t` is an array, the
            probability is computed for each element and an array of the same
            shape is returned.

        Notes:
            The probability is non-zero only if `beta < 1`.
        """
        if self.beta < 1:
            if self.r == 0:
                return special.gammaincc(
                    0.5/(1-self.beta),
                    self.s**(2*(1-self.beta)) /
                    (2*(self.sigma*(1-self.beta))**2*t))
            return special.gammaincc(
                0.5/(1-self.beta),
                self.r*self.s**(2*(1-self.beta)) /
                (self.sigma**2*(1-self.beta) *
                 (1-np.exp(2*self.r*(self.beta-1)*t))))
        return 0

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
            If `t` and `k` are scalars, returns the price of a call option as a
            scalar value. If `t` and/or `k` are arrays, applies NumPy
            broadcasting rules and returns an array of prices.

        Notes:
            For computation on a grid, use `call_price(*vol_grid(t, k))`, where
            `t` and `k` are 1-D arrays with grid coordinates.
        """
        # TODO Shall we select tolerance more accurately?
        if np.isclose(self.beta, 1, rtol=0, atol=1e-12):
            return blackscholes.BlackScholes(
                self.s, self.sigma, self.r).call_price(t, k)

        nu = 0.5/(self.beta-1)
        t_adj, k_adj = self._remove_drift(t, k)
        xi = self.s**(2*(1-self.beta)) / ((self.sigma*(1-self.beta))**2*t_adj)
        y = k_adj**(2*(1-self.beta)) / ((self.sigma*(1-self.beta))**2*t_adj)

        if self.beta < 1:
            return (self.s*st.ncx2(2*(1-nu), xi).sf(y) -
                    k_adj*st.ncx2(-2*nu, y).cdf(xi))
        else:
            return (self.s*st.ncx2(2*nu, y).sf(xi) -
                    k_adj*st.ncx2(2*(1+nu), xi).cdf(y))

    def iv(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]],
        use_approx: bool = False
    ) -> Union[float, NDArray[float_]]:
        """Computes the Black-Scholes implied volatility produced by the model.

        This function first computes the price of a call option with expiration
        time `t` and strike `k`, and then inverts the Black-Scholes formula to
        find `sigma`.

        Args:
            t: Expiration time (float or ndarray).
            k: Strike (float or ndarray).
            use_approx: If True, uses Hagan and Woodward's approximate formula
                to compute option prices. If False, uses the exact formula.

        Returns:
            If `t` and `k` are scalars, returns a scalar value. If `t` and/or
            `k` are arrays, applies NumPy broadcasting rules and returns an
            array. If the implied volatility cannot be computed (i.e. cannot
            solve the Black-Scholes formula for `sigma`), returns NaN in the
            scalar case or puts NaN in the corresponding cell of the array.
        """
        if np.isclose(self.beta, 1, rtol=0, atol=1e-12):
            if isinstance(k, float):
                return self.sigma
            else:
                return self.sigma*np.ones_like(k)

        if not use_approx:
            return blackscholes.call_iv(
                self.s, self.r, self.call_price(t, k), t, k)
        else:
            t_adj, k_adj = self._remove_drift(t, k)
            f = 0.5*(self.s + k_adj)
            return np.sqrt(t_adj/t)*(
                self.sigma*f**(self.beta-1) *
                (1 + (1-self.beta)*(2+self.beta)/24*((self.s-k_adj)/f)**2 +
                 (1-self.beta)**2/24*self.sigma**2*t_adj*f**(2*(self.beta-1))))

    @staticmethod
    def _calibrate_objective(
        x: tuple[float, float],
        s: float,
        r: float,
        t: Union[float, NDArray[float_]],
        k: NDArray[float_],
        iv: NDArray[float_],
        use_approx: bool
    ) -> float:
        """The objective function for parameter calibration.

        Computes the sum of squares of differences between the model and market
        implied volatilities, for each expiration time and strike.

        Args:
            x: Tuple of calibrated parameters `(sigma, beta)`.
            s: Initial price.
            r: Interest rate.
            t: Array of expiration times.
            k: Array of strikes.
            iv: Array of market implied volatilities for each combination of
            expiration time and strike, i.e. with shape `(len(t), len(k))`. May
            contain NaNs, which are ignored when computing the sum of squares.
            use_approx: If True, uses Hagan and Woodward's approximate formula.
        """
        C = CEV(s=s, sigma=x[0], beta=x[1], r=r)
        return sum((C.iv(t[i], k[i], use_approx) - iv[i])**2
                   for i in range(len(iv)))

    @classmethod
    def calibrate(
        cls,
        t: Union[float, NDArray[float_]],
        k: NDArray[float_],
        iv: NDArray[float_],
        s: float,
        r: float = 0,
        use_approx: bool = False,
        min_method: str = "L-BFGS-B",
        beta0=0.8,
        return_minimize_result: bool = False
    ):
        """Calibrates the parameters of the CEV model.

        This function finds the parameters `sigma` and `beta` of the model
        which minimize the sum of squares of the differences between market
        and model implied volatilities. Returns an instance of the class with
        the calibrated parameters.

        Args:
            t : Expiration time (scalar or array).
            k: Array of strikes.
            iv: Array of market implied volatilities.
            s: Initial price.
            r: Interest rate.
            use_approx: If True, uses Hagan and Woodward's approximate formula.
            min_method: Minimization method to be used, as accepted by
                `scipy.optimize.minimize`. The method must be able to handle
                bounds.
            beta0: Initial guess of parameter `beta` (see notes below).
            return_minimize_result: If True, return also the minimization
                result of `scipy.optimize.minimize`.

        Returns:
            If `return_minimize_result` is True, returns a tuple `(cls, res)`,
            where `cls` is an instance of the class with the calibrated
            parameters and `res` is the optimization result returned by
            `scipy.optimize.minimize`. Otherwise returns only `cls`.

        Notes:
            It is advised not to set `beta0=1`, since for `beta` close to 1
            (currently, for `beta` in `[1-1e-12, 1+1e-12]`) the model
            internally switches to the Black-Scholes formula for computation of
            option prices, and there is risk that the minimization method will
            get stuck at `beta=1`.
        """
        k_ = k.flatten()
        iv_ = iv.flatten()
        if isinstance(t, float):
            t_ = np.ones_like(k_)*t
        else:
            t_ = t.flatten()
        sigma0 = iv_[np.abs(k_-s).argmin()]

        res = optimize.minimize(
            fun=CEV._calibrate_objective,
            x0=(sigma0, beta0),
            bounds=[(0, np.Inf), (0, np.Inf)],
            args=(s, r, t_, k_, iv_, use_approx),
            method=min_method)
        ret = cls(s=s, sigma=res.x[0], beta=res.x[1], r=r)
        if return_minimize_result:
            return ret, res
        else:
            return ret

    def simulate(
        self,
        t: float,
        steps: int,
        paths: int
    ) -> NDArray[float_]:
        """Simulates paths of the price process.

        This function uses the exact transitional distribution for simulation.

        Args:
            t: Time horizon.
            steps: Number of simulation points minus 1, i.e. paths are sampled
                at `t_i = i*dt`, where `i = 0, ..., steps`, `dt = t/steps`.
            paths: Number of paths to simulate.

        Returns:
            An array `s` of shape `(steps+1, paths)`, where `s[i, j]` is the
            value of `j`-th path at point `t_i`.

        Notes:
            When `beta` of the model is greater or equal than 1, the simulation
            is fast thanks to NumPy's vectorization. For `beta < 1`, it is much
            slower.
        """
        dt = t/steps
        # First simulate X, a squared Bessel process of dimension `delta`.
        # We simulate it at points `tau(t_i)`, then get the CEV process with
        # zero drift at points `tau(t_i)`, and finally get the process with the
        # desired drift.
        X = np.empty(shape=(steps+1, paths))
        X[0] = self.s**(2*(1-self.beta)) / (self.sigma*(1-self.beta))**2
        delta = (1-2*self.beta) / (1-self.beta)

        if self.beta < 1:
            for i in range(steps):
                dtau = (dt if self.r == 0
                        else (self._remove_drift((i+1)*dt, 0)[0] -
                              self._remove_drift(i*dt, 0)[0]))
                u = st.uniform.rvs(size=paths)
                # TODO Is it possible to vectorize this?
                for j in range(paths):
                    if (X[i, j] == 0 or u[j] <= st.chi2.cdf(
                            X[i, j], df=2-delta, scale=dtau)):
                        X[i+1, j] = 0
                    else:
                        X[i+1, j] = optimize.newton(
                            lambda x: u[j] - st.ncx2.sf(X[i, j], df=2-delta,
                                                        nc=x/dtau, scale=dtau),
                            x0=X[i, j])
        else:
            for i in range(steps):
                dtau = (dt if self.r == 0
                        else (self._remove_drift((i+1)*dt, 0)[0] -
                              self._remove_drift(i*dt, 0)[0]))
                X[i+1] = st.ncx2(
                    df=delta, nc=X[i]/dtau, scale=dtau).rvs(size=paths)

        # The process with zero drift at points `tau(t_i)`
        S0 = ((self.sigma*self.beta)**2*X)**(-0.5/self.beta)

        if self.r == 0:
            return S0
        else:
            return S0 * np.exp(
                self.r*np.arange(0, steps+1).reshape((steps+1, 1))*dt)
