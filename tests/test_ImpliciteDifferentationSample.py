from dbopt.ImpliciteDifferentationSample import ImpliciteDifferentationSample

import jax.numpy as jnp

def test_implicite_differentiation():
    t = jnp.array([1., 2., 3.])
    implicite_differentiation = ImpliciteDifferentationSample()
    print(implicite_differentiation.implicite_differentiation(t))
    assert jnp.allclose(implicite_differentiation.implicite_differentiation(t),
                        jnp.array([1., 1., 1.]))