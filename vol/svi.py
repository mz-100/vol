from typing import Optional
from dataclasses import dataclass
import numpy as np
from scipy import optimize  # type: ignore
from volatility.misc import FloatArray, Floats


@dataclass
class SVI:
    """The SVI (Stochastic Volatility Inspired) model.

    The model directly represents a volatility curve by the function
      `w(x) = a + b*(rho*(x-m) + sqrt((x-m)**2 + sigma**2))`
    where
      `x` is the log-moneyness, i.e. `x = log(s/k)`,
      `w` is the total implied variance, i.e. `w = t*(iv**2)`,
    and `a`, `b >= 0`, `-1 < rho < 1`, `m`, `sigma > 0` are model parameters.
    Expiration time is assumed to be fixed and is not explicitly specified (but
    see `to_jumpwing` and `from_jumpwing` functions).

    The above formula is the so-called 'raw' parametrization. The class also
    provides function for conversion to/from 'natural' and 'jumpwing'
    parametrizations.

    Attributes:
      a, b, rho, m, sigma: Model parameters.

    Methods:
      to_natural: Returns the natural parameters.
      from_natural: Constructs the model from natural parameters.
      to_jumpwing: Returns the jump-wing parameters.
      from_jumpwing: Constructs the model from jump-wing parameters.
      calibrate: Calibrates parameters of the model.
      durrleman_function: Computes Durrleman's function.
      durrleman_condition: Find the minimum of Durrleman's function.
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_natural(self) -> tuple[float, float, float, float, float]:
        """Returns parameters of the natural parametrization.

        The natural parametrization is:
          `w(x) = delta + omega/2 * (1 + zeta*rho*(x-mu)
                + sqrt((zeta*(x-mu) + rho)**2 + 1 - rho**2)`

        Returns:
          Tuple `(delta, mu, rho, omega, zeta)`.
        """
        omega = 2*self.b*self.sigma / np.sqrt(1-self.rho**2)
        delta = self.a - 0.5*omega*(1-self.rho**2)
        mu = self.m + self.rho*self.sigma / np.sqrt(1-self.rho**2)
        zeta = np.sqrt(1-self.rho**2) / self.sigma
        return delta, mu, self.rho, omega, zeta

    @classmethod
    def from_natural(cls, delta: float, mu: float, rho: float, omega: float,
                     zeta: float):
        """Construct a class instance from natural parameters.

        See `to_natural` method for the formula of this parametrization.
        """
        return cls(a=delta+0.5*omega*(1-rho**2),
                   b=0.5*omega*zeta,
                   rho=rho,
                   m=mu-rho/zeta,
                   sigma=np.sqrt(1-rho**2)/zeta)

    def jumpwing(
            self, t: float = 1) -> tuple[float, float, float, float, float]:
        """Returns parameters of the jump-wing parametrization.

        This parametrization depends on the time to expiration of the options
        constituting the volatility curve. See Gatheral's presentation (2004)
        for details and formulas.

        Args:
          t: Time to expiration of the options.

        Returns:
          Tuple  `(v, psi, p, c, v_tilde)` of the jump-wing parameteres.
        """
        w = (self.a +
             self.b*(-self.rho*self.m + np.sqrt(self.m**2+self.sigma**2)))
        v = w/t
        psi = self.b/np.sqrt(w)/2 * (
            -self.m/np.sqrt(self.m**2+self.sigma**2) + self.rho)
        p = self.b*(1-self.rho)/np.sqrt(w)
        c = self.b*(1+self.rho)/np.sqrt(w)
        v_tilde = (self.a + self.b*self.sigma*np.sqrt(1-self.rho**2)) / t
        return v, psi, p, c, v_tilde

    @classmethod
    def from_jumpwing(cls, v: float, psi: float, p: float, c: float,
                      v_tilde: float, t: float = 1):
        """Construct a class instance from jump-wing parameters."""
        w = v*t
        b = 0.5*np.sqrt(w)*(c+p)
        rho = 1 - 2*p/(c+p)
        beta = rho - 2*psi*np.sqrt(w)/b
        if np.abs(beta) > 1:
            raise ValueError(
                f"Smile is not convex: beta={beta}, but must be in [-1, 1].")
        elif beta == 0:
            m = 0
            sigma = (v-v_tilde)*t / (b*(1-np.sqrt(1-rho**2)))
        else:
            alpha = np.sign(beta) * np.sqrt(1/beta**2 - 1)
            m = (v-v_tilde)*t / (b*(-rho+np.sign(alpha)*np.sqrt(1+alpha**2) -
                                    alpha*np.sqrt(1-rho**2)))
            sigma = alpha*m
        a = v_tilde*t - b*sigma*np.sqrt(1-rho**2)
        return cls(a, b, rho, m, sigma)

    def __call__(self, x: FloatArray) -> FloatArray:
        """Returns the total implied variance `w(x)`."""
        return self.a + self.b*(self.rho*(x-self.m) +
                                np.sqrt((x-self.m)**2 + self.sigma**2))

    def durrleman_function(self, x: Floats) -> Floats:
        """Durrleman's function for verifying the convexity of a price surface.

        Args:
          x: Log-moneyness (scalar or array).

        Returns:
          The value of Durrleman's function. If `x` is an array, then an array
          of the same shape is returned.
        """
        # Total variance and its two derivatives
        w = self.__call__(x)
        wp = self.b*(self.rho + (x-self.m) / np.sqrt(
            (x-self.m)**2 + self.sigma**2))
        wpp = self.b*(1/np.sqrt((x-self.m)**2 + self.sigma**2) -
                      (x-self.m)**2/((x-self.m)**2 + self.sigma**2)**1.5)
        return (1-0.5*x*wp/w)**2 - 0.25*wp**2*(1/w+0.25) + 0.5*wpp

    def durrleman_condition(
            self,
            min_x: Optional[float] = None,
            max_x: Optional[float] = None) -> tuple[bool, float]:
        """Checks Durrleman's condition.

        This function tries to find numerically the global minimum of
        Durrleman's function (if this minimum is negative, then Durrleman's
        condition fails, so the model has static arbitrage). The Dual Annelaing
        minimization method is used.

        Args:
          min_x, max_x: Specify the interval on which Durrleman's function is
            minimized. `None` corresponds to the plus or minus infinity.

        Returns:
          Tuple `(min, x)` where `min` is the minimum of Durrleman's function,
            and `x` is the point where it is attained.
        """
        res = optimize.dual_annealing(
            lambda x: self.durrleman_function(x[0]), x0=[0],
            bounds=[(min_x, max_x)],
            minimizer_kwargs={
                "method": "BFGS",
                "jac": (lambda x:
                        self.b*(self.rho + (x[0]-self.m) /
                                np.sqrt((x[0]-self.m)**2+self.sigma**2)))
            })
        return res.fun, res.x[0]

    @staticmethod
    def _calibrate_adc(x: FloatArray, w: FloatArray, m: float,
                       sigma: float) -> float:
        """Calibrates the raw parameters `a, d, c` given `m, sigma`.

        This is an auxiliary function used in the two-step calibration
        procedure. It finds `a, d, c` which minimize the sum of squares of the
        differences of the given total implied variances and the ones produced
        by the model, assuming that `m, sigma` are given and fixed.

        Args:
          x: Array of log-moneynesses
          w: Array of total implied variances.
          m: Parameter `m` of the model.
          sigma: Parameter `sigma` of the model.

        Returns:
          Tuple `((a, d, c), f)` where `a, d, c` are the calibrated parameters
          and `f` is the value of the objective function at the minimum.
        """
        # Objective function; p = (a, d, c)
        def f(p):
            return 0.5*np.linalg.norm(
                p[0] + p[1]*(x-m)/sigma + p[2]*np.sqrt(((x-m)/sigma)**2+1) -
                w)**2

        # Gradient of the objective function
        def fprime(p):
            v1 = (x-m)/sigma
            v2 = np.sqrt(((x-m)/sigma)**2+1)
            v = p[0] + p[1]*v1 + p[2]*v2 - w
            return (np.sum(v), np.dot(v1, v), np.dot(v2, v))

        res = optimize.minimize(
            f,
            x0=(np.max(w)/2, 0, 2*sigma),
            method="SLSQP",
            jac=fprime,
            bounds=[(None, np.max(w)), (None, None), (0, 4*sigma)],
            constraints=[
                {'type': 'ineq',
                 'fun': lambda p: p[2]-p[1],
                 'jac': lambda _: (0, -1, 1)},
                {'type': 'ineq',
                 'fun': lambda p: p[2]+p[1],
                 'jac': lambda _: (0, 1, 1)},
                {'type': 'ineq',
                 'fun': lambda p: 4*sigma - p[2]-p[1],
                 'jac': lambda _: (0, -1, -1)},
                {'type': 'ineq',
                 'fun': lambda p: p[1]+4*sigma-p[2],
                 'jac': lambda _: (0, 1, -1)}])
        return res.x, res.fun

    @classmethod
    def calibrate(cls,
                  x: FloatArray,
                  w: FloatArray,
                  min_sigma: float = 1e-4,
                  max_sigma: float = 10,
                  return_minimize_result: bool = False):
        """Calibrates the parameters of the model.

        This function finds the parameters which minimize the sum of squares of
        the differences of the given total implied variances and the ones
        produced by the model.

        The two-step minimization procedure is used (by Zeliade Systems, see
        their white paper). For each pair of parameters `sigma, m`, parameters
        `a, d, c` are found by using a gradient method; then `sigma, m` are
        found by a stochastic method (namely, SLSQP and Dual Annealing are
        used).

        Args:
          x: Array of log-moneynesses
          w: Array of total implied variances.
          min_sigma, max_sigma: Bounds for `sigma` parameter.
          return_minimize_result: If True, return also the minimization result
            of `sciy.optimize.dual_annealing`.

        Returns:
          If `return_minimize_result` is True, returns a tuple `(cls, res)`,
          where `cls` is an instance of the class with the calibrated
          parameters and `res` in the optimization result returned by
          `scipy.optimize.dual_annealing`. Otherwise returns only `cls`.
        """
        res = optimize.dual_annealing(
            lambda q: cls._calibrate_adc(x, w, q[0], q[1])[1],  # q=(m, sigma)
            bounds=[(min(x), max(x)), (min_sigma, max_sigma)],
            minimizer_kwargs={"method": "nelder-mead"})
        m, sigma = res.x
        a, d, c = cls._calibrate_adc(x, w, m, sigma)[0]
        rho = d/c
        b = c/sigma
        ret = cls(a, b, rho, m, sigma)
        if return_minimize_result:
            return ret, res
        else:
            return ret


def vogt_example() -> SVI:
    """Returns an SVI parametrization not satisfying Durrleman's condition.

    See Gatheral, Jacquier "Arbitrage-free SVI volatility surfaces" (2014),
    Example 3.1 for details.
    """
    return SVI(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
