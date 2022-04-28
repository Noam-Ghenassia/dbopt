import pytest
import jax.numpy as jnp
from dbopt.DB_Top_opt import DB_Top_opt
from dbopt.Bumps import Bumps
from dbopt.DB_sampler import DB_sampler

__author__ = "Noam Ghenassia"
__copyright__ = "Noam Ghenassia"
__license__ = "MIT"

bumps = Bumps()
sampler = DB_sampler()
n_sampling = 20
opt = DB_Top_opt(bumps.level, n_sampling)

def test_get_points():
    init_points = opt.get_points()
    sampling_loss = sampler._loss(init_points, theta=0, net=bumps.level)
    assert jnp.allclose(sampling_loss, 0)


    
# TODO: add circle test


# TODO: add bump test with computed gradient (!only the gradient) of topological loss