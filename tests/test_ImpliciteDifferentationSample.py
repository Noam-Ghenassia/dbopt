from dbopt.ImpliciteDifferentationSample import ImpliciteDifferentationSample

from jax import grad
import jax.numpy as jnp
from jaxopt.implicit_diff import custom_root

def test_implicite_differentiation():
    t = jnp.array([1., 2., 3.])
    ids = ImpliciteDifferentationSample()
    grads = grad(lambda t: 
                custom_root(ids._optimality_condition)
                (ids._inner_problem)(None, t).sum()
            )(t)
    assert jnp.allclose(grads,
                        jnp.array([1., 1., 1.]))
# %%
