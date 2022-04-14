from jax import numpy as jnp
from jax import grad
from jaxopt import implicit_diff


class ImpliciteDifferentationSample:
    """Compute the derivative of the argmin x_star(t) of the equation
    (x - t)**2 with respect to t using the implicit differentiation
    technique.
    """

    def __init__(self):
        pass

    def _optimality_condition(self, x_star, t):
        """Optimality condition for x_star(t): x_star(t) - t = 0.

        Args:
            t (jnp.array): the point at which we compute the derivative
            x_star (jnp.array): the point at which we compute the derivative

        Returns:
            jnp.array: the value of the optimality condition
        """
        return x_star - t

    @implicit_diff.custom_root(_optimality_condition)
    def _inner_problem(self, t):
        """Inner optimization problem: argmin x_star(t) of the equation
        (x - t)**2.

        Args:
            t (jnp.array): the point at which we compute the derivative

        Returns:
            jnp.array: the value of the inner optimization problem
        """
        return t
    
    def implicite_differentiation(self, t):
        """Compute the derivative of the argmin x_star(t) of the equation
        (x - t)**2 with respect to t using the implicit differentiation
        technique.

        Args:
            t (jnp.array): the point at which we compute the derivative

        Returns:
            jnp.array: the value of the derivative
        """
        x_star = self._inner_problem(t)
        return grad(x_star)(t)
