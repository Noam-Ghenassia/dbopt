from dbopt.ImpliciteDifferentiationSample import ImpliciteDifferentationSample

from jax import grad
import jax.numpy as jnp
from jaxopt.implicit_diff import custom_root

def test_implicite_differentiation():
    t = jnp.array([1., 2., 3.])
    ids = ImpliciteDifferentationSample()

    sum = lambda a: jnp.sum(a)

    grads = grad(lambda t: sum(ids.x_star(t)))(t)
    assert jnp.allclose(grads,
                        jnp.array([1., 1., 1.]))