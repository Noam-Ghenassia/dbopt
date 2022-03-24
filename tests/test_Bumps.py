import pytest
from dbopt.Bumps import Bumps
import jax.numpy as jnp

__author__ = "Noam Ghenassia"
__copyright__ = "Noam Ghenassia"
__license__ = "MIT"


def test_level():
    bumps = Bumps()
    x = jnp.array([[-1, 0], [1, 0], [0, 0], [0, 1]])
    y = jnp.array([jnp.exp(-6)+1, jnp.exp(-6)+1, 2*jnp.exp(-1.5), 2*jnp.exp(-3)])
    assert jnp.all(jnp.equal(bumps.level(x), y))

    assert jnp.all(jnp.equal(bumps.level(x, 1), y-1))
    