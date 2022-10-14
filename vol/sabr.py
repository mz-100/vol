from typing import Union
from numpy import float_
from numpy.typing import NDArray
from dataclasses import dataclass
import numpy as np
import scipy.stats as st         # type: ignore
import scipy.optimize as opt     # type: ignore
from . import blackscholes


@dataclass
class SABR:
    """The SABR (Stochastic Alpha, Beta, Rho) model.

    The base asset is stock which under the pricing measure follows the SDEs
        ```
        d(S_t) = alpha_t * S_t^beta * d(W^1_t),
        d(alpha_t) = nu * alpha_t * d(W^2_t),
        ```
    where `W^1_t` and `W^2_t` are standard Brownian motions with correlation
    coefficient `rho`, and `alpha > 0, beta > 0, nu >= 0` are other model
    parameters.

    Attributes:
        s: Initial price, i.e. S_0.
        alpha, beta, rho, nu: Model parameters.

    Methods:
        call_price: Computes the (approximate) price of a call option.
        iv: Computes the approximate implied volatility by Hagan's formula.
        calibrate: Calibrates parameters of the model.
        simulate: Simulates paths by Euler's scheme.

    Notes:
        The risk-free interest rate is not specified (as SABR typically models
        forward prices). If you need it, e.g. to discount option prices, pass
        the corresponding parameter to the pricing function.
    """
    s: float
    alpha: float
    beta: float
    rho: float
    nu: float

    def call_price(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]],
        discount: Union[float, NDArray[float_]] = 1
    ) -> Union[float, NDArray[float_]]:
        """Computes the (approximate) price of a call option.

        This function approximately computes
            `discount * E(S_t-k)^+`
        where `S_t` is the SABR price process. The approximation is done by
        computing the implied volatility by Hagan's formula, and then applying
        Black's pricing formula (with zero interest rate).

        The `discount` coefficient depends upon whether options on futures or
        forward contracts are considered, i.e. it will be `exp(-r*t)` for a
        futures option (`r` is the interest rate), or `exp(-r*T)` for a forward
        option (`T` is the forward contract expiration).

        Args:
            t: Expiration time.
            k: Strike.
            discount: Discount coefficient.

        Returns:
            If `t`, `k`, `discount` are scalars, returns the price of a call
            option as a scalar value. If they are arrays, applies NumPy
            broadcasting rules and returns the array of prices.
        """
        return (discount * blackscholes._call_price(
            s=self.s, sigma=self.iv(t, k), t=t, k=k))

    def iv(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """Hagan's approximate formula for Black's implied volatility.

        Args:
            t: Expiration time.
            k: Strike.

        Returns:
            If `t`, `k` are scalars, returns the implied volatility as a scalar
            value. If they are arrays, applies NumPy broadcasting rules and
            returns an array of implied volatilities.

        Notes:
            For general `beta`, the second order approximation in the
            perturbation expansion is used. For `beta=0` or `beta=1`, the
            fourth order approximation is used (see Hagan et al.'s paper).
        """
        if self.beta == 0:
            z = self.nu/self.alpha * np.sqrt(self.s*k) * np.log(self.s/k)
            x = np.log((np.sqrt(1-2*self.rho*z + z*z) + z - self.rho) /
                       (1-self.rho))
            return (
                self.alpha *
                # when f = k, we must have log(f/k)/(f-k) = k, z/x = 1
                np.divide(np.log(self.s/k)*z, (self.s-k)*x,
                          where=np.abs(self.s-k) > 1e-12,
                          out=np.array(k, dtype=float)) *
                (1 + t*(self.alpha**2/(24*self.s*k) +
                        (2-3*self.rho**2)/24*self.nu**2)))
        elif self.beta == 1:
            z = self.nu/self.alpha * np.log(self.s/k)
            x = np.log((np.sqrt(1-2*self.rho*z + z*z) + z - self.rho) /
                       (1-self.rho))
            return (
                self.alpha *
                # when f = k, we must have z/x = 1
                np.divide(z, x,
                          where=np.abs(self.s-k) > 1e-12,
                          out=np.ones_like(z)) *
                (1 + t*(self.rho*self.alpha*self.nu/4 +
                        (2-3*self.rho**2)*self.nu**2/24)))
        else:
            z = (self.nu/self.alpha *
                 (self.s*k)**((1-self.beta)/2) * np.log(self.s/k))
            x = np.log((np.sqrt(1-2*self.rho*z + z*z) + z -
                        self.rho)/(1-self.rho))
            return (
                self.alpha /
                (self.s*k)**((1-self.beta)/2)*(
                    1 +
                    (1-self.beta)**2/24*np.log(self.s/k)**2 +
                    (1-self.beta)**4/1920*np.log(self.s/k)**4) *
                # when f = k, we must have z/x = 1
                np.divide(z, x, where=np.abs(z) > 1e-12, out=np.ones_like(z)) *
                (1 + t*(
                    ((1-self.beta)**2/24 * self.alpha**2 /
                     (self.s*k)**(1-self.beta)) +
                    (self.rho*self.beta*self.nu*self.alpha /
                     (4*(self.s*k)**((1-self.beta)/2))) +
                    (2-3*self.rho**2)/24*self.nu**2)))

    @classmethod
    def calibrate(cls,
                  t: Union[float, NDArray[float_]],
                  k: NDArray[float_],
                  iv: NDArray[float_],
                  s: float,
                  calibrate_beta: bool = True,
                  beta0: float = 1,
                  min_method: str = "SLSQP",
                  return_minimize_result: bool = False):
        """Calibrates the parameters of the SABR model.

        This function finds the parameters `alpha`, `beta`, `rho`, `nu` of the
        model which minimize the sum of squares of the differences between
        market and model implied volatilities. Returns an instance of the class
        with the calibrated parameters.

        Args:
            t : Expiration time (scalar or array).
            k: Array of strikes.
            iv: Array of market implied volatilities.
            s: Initial price.
            calibrate_beta: If False, does not calibrate `beta`, but sets it
                equal to `beta0`.
            beta0: If `calibrate_beta` is True, `beta0` is used as the initial
                guess of `beta` parameter.
            min_method: Minimization method to be used, as accepted by
                `scipy.optimize.minimize`. The method must be able to handle
                bounds.
            return_minimize_result: If True, return also the minimization
                result of `scipy.optimize.minimize`.

        Returns:
            If `return_minimize_result` is True, returns a tuple `(cls, res)`,
            where `cls` is an instance of the class with the calibrated
            parameters and `res` in the optimization result returned by
            `scipy.optimize.minimize`. Otherwise returns only `cls`.
        """
        alpha0 = iv[np.abs(k-s).argmin()]  # ATM volatility

        if calibrate_beta:
            res = opt.minimize(
                fun=lambda p: np.linalg.norm(SABR(s, *p).iv(t, k) - iv),
                x0=(alpha0, beta0, 0, 1),  # (alpha, beta, rho, nu)
                method=min_method,
                bounds=[(0, np.Inf), (0, np.Inf), (-1, 1), (0, np.Inf)])
            ret = cls(s=s, alpha=res.x[0], beta=res.x[1], rho=res.x[2],
                      nu=res.x[3])
        else:
            res = opt.minimize(
                fun=lambda p: np.linalg.norm(
                    SABR(s, p[0], beta0, p[1], p[2]).iv(t, k) - iv),
                x0=(alpha0, 0, 1),  # (alpha, rho, nu)
                method=min_method,
                bounds=[(0, np.Inf), (-1, 1), (0, np.Inf)])
            ret = cls(s=s, alpha=res.x[0], beta=beta0, rho=res.x[1],
                      nu=res.x[2])
        if return_minimize_result:
            return ret, res
        else:
            return ret

    def simulate(
            self,
            t: float,
            steps: int,
            paths: int,
            return_alpha: bool = False
            ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Simulates paths using Euler's scheme.

        Args:
            t: Time interval.
            steps: Number of simulation points minus 1, i.e. paths are sampled
                at `t_i = i*dt`, where `i = 0, ..., steps`, `dt = t/steps`.
            paths : Number of paths to simulate.
            return_alpha : If True, returns both price and volatility
                processes.

        Returns:
            If `return_alpha` is False, returns an array `s` of shape
            `(paths, steps+1)`, where `s[i, j]` is the value of `j`-th path of
            the price process at point `t_i`.

            If `return_alpha` is True, returns a tuple `(s, a)`, where `s` and
            `a` are arrays of shape `(paths, steps+1)` representing the price
            and volatility processes.

        Notes:
            Depending on the value of the model's beta, we use the following
            variations of the Euler scheme, which are different in how the
            price process is obtained.
            * beta = 0
                The price process is the stochastic integral of the volatility
                process. The integrand is approximated at the left point, the
                process may cross zero and become negative.
            * 0 < beta < 1
                The standard Euler's scheme is used. A path can be trapped at
                zero, and in this case it stays at zero until the end.
            * beta >= 1
                The log-price is simulated (and then exponentiated). The
                resulting process is strictly positive.

            In the cases `beta=0` and `beta=1`, NumPy's vectorization is
            applied to whole arrays of paths, while in the other cases, it is
            applied step-wise (which makes simulation slower).
        """

        dt = t/steps
        sqrtdt = np.sqrt(dt)
        Z = st.norm.rvs(size=(2, steps, paths))
        Alpha = self.alpha*np.exp(np.concatenate([
            np.zeros(shape=(1, paths)),
            np.cumsum(Z[0]*self.nu*sqrtdt - 0.5*self.nu**2*dt, axis=0)]))

        if self.beta == 0:
            S = np.cumsum(np.concatenate([
                np.ones(shape=(1, paths))*self.s,
                Alpha[:-1] *
                (self.rho*Z[0] + np.sqrt(1-self.rho**2)*Z[1])*sqrtdt]), axis=0)
        elif self.beta < 1:
            S = np.empty_like(Alpha)
            S[0] = self.s
            for i in range(steps):
                S[i+1] = np.maximum(
                    0,
                    S[i] + Alpha[i]*S[i]**self.beta *
                    (self.rho*Z[0, i] + np.sqrt(1-self.rho**2)*Z[1, i])*sqrtdt)
        elif self.beta == 1:
            X = np.cumsum(np.concatenate([
                np.zeros(shape=(1, paths)),
                -0.5*Alpha[:-1]**2*dt + Alpha[:-1] *
                (self.rho*Z[0] + np.sqrt(1-self.rho**2)*Z[1])*sqrtdt]), axis=0)
            S = self.s*np.exp(X)
        else:
            X = np.empty_like(Alpha)
            X[0] = np.log(self.s)
            for i in range(steps):
                X[i+1] = X[i] - (
                    0.5*Alpha[i]**2*np.exp(2*(self.beta-1)*X[i])*dt +
                    Alpha[i]*np.exp((self.beta-1)*X[i]) *
                    (self.rho*Z[0, i] + np.sqrt(1-self.rho**2)*Z[1, i])*sqrtdt)
            S = np.exp(X)
        if return_alpha:
            return S, Alpha
        else:
            return S
