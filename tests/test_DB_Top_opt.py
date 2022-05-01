import pytest
import jax.numpy as jnp
from jax import grad
from dbopt.DB_Top_opt import DecisionBoundrayGradient
from dbopt.Bumps import Bumps
from dbopt.DB_sampler import DB_sampler

__author__ = "Noam Ghenassia"
__copyright__ = "Noam Ghenassia"
__license__ = "MIT"

#bumps = Bumps()
#sampler = DB_sampler()
#n_sampling = 20
#opt = DB_Top_opt(bumps.level, n_sampling)

"""def test_get_points():
    init_points = opt.get_points()
    sampling_loss = sampler._loss(init_points, theta=0, net=bumps.level)
    assert jnp.allclose(sampling_loss, 0)"""


    
def test_grad_of_unit_circle_sampling():
    # t_star is the function that gives the parameters t such that the new points
    # lie on the circle of radius r. there are 2 points, and changing the value
    # of r (making it bigger) changes the error of each point by a factor 1.
    net = lambda x, r: (x ** 2).sum(axis=1) - r ** 2 * jnp.ones(x.shape[0])
    r = jnp.array([1.0])
    x1 = jnp.array([[1, 1], [2, 0]])
    x2 = jnp.array([[]])
    x1_normalized = x1 / jnp.sqrt((x1**2).sum(axis=1)).reshape(-1, 1)
    db_opt = DecisionBoundrayGradient(net, x1_normalized)
    assert jnp.allclose(grad(lambda r: db_opt.t_star(r).sum())(r), 2.)
    x2 = jnp.array([[1, 1], [-1, 1], [2, 0]])
    x2_normalized = x2 / jnp.sqrt((x2**2).sum(axis=1)).reshape(-1, 1)
    db_opt = DecisionBoundrayGradient(net, x2_normalized)
    assert jnp.allcose(grad(lambda r: db_opt.t_star(r).sum())(r), 3.)
    


# TODO: add bump test with computed gradient (!only the gradient) of topological loss