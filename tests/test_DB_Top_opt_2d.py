import pytest
import jax.numpy as jnp
from jax import grad, jacrev
from dbopt.DB_Top_opt_2d import DecisionBoundrayGradient
from dbopt.DB_Top_opt_2d import SingleCycleDecisionBoundary
from dbopt.DB_Top_opt_2d import SingleConnectedComponent
from dbopt.Bumps import Bumps
from dbopt.DB_sampler_2d import DecisionBoundarySampler

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
    net = lambda x, r: (x ** 2).sum(axis=1) - r **2 * jnp.ones(x.shape[0])
    r = jnp.array([3.0])
    
    x1 = jnp.array([[1, 1], [2, 0]])
    x1_normalized = 3*(x1 / jnp.sqrt((x1**2).sum(axis=1)).reshape(-1, 1))
    db_opt = DecisionBoundrayGradient(net, x1_normalized)
    assert jnp.allclose(grad(lambda r: db_opt.t_star(r).sum())(r), 2.)
    
    x2 = jnp.array([[1, 1], [-1, 1], [1, 2], [2, 1], [0, -1]])
    x2_normalized = 3*(x2 / jnp.sqrt((x2**2).sum(axis=1)).reshape(-1, 1))
    db_opt = DecisionBoundrayGradient(net, x2_normalized)
    assert jnp.allclose(grad(lambda r: db_opt.t_star(r).sum())(r), 5.0)
    assert jnp.allclose(jacrev(lambda r:  db_opt.t_star(r))(r), jnp.array([[1.], [1.], [1.], [1.], [1.]]))

def test_grad_of_diagonal_line_sampling():
    net = lambda x, r: x.sum(axis=1)-r*jnp.ones(x.shape[0])
    r = jnp.array([3.])
    x = jnp.array([[1., 2.], [2., 1.], [4., -1.], [6., -3.]])
    db_opt = DecisionBoundrayGradient(net, x)
    assert jnp.allclose(grad(lambda r: db_opt.t_star(r).sum())(r), 4/jnp.sqrt(2))
    assert jnp.allclose(jacrev(lambda r: db_opt.t_star(r))(r), (1/jnp.sqrt(2))*jnp.array([[1.], [1.], [1.], [1.]]))

def test_single_cycle_loss_4_points():
    net = lambda x, r: (x ** 2).sum(axis=1) - r **2 * jnp.ones(x.shape[0])
    x = jnp.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    r = jnp.array([jnp.sqrt(2)])
    sc = SingleCycleDecisionBoundary(net, x)
    assert jnp.allclose(sc.differentiable_topological_loss(r), -2*(2-jnp.sqrt(2))**2)
    assert jnp.allclose(grad(lambda r: sc.differentiable_topological_loss(r))(r), -2*jnp.sqrt(2)*(2-jnp.sqrt(2))**2)
    
    
def test_one_connected_component_with_bumps():
    bumps = Bumps()
    theta = jnp.array([0.])
    net = lambda x, theta: bumps.level(x, theta)
    sampling = jnp.array([[-0.5325, 0.], [0.5325, 0]])
    sc = SingleConnectedComponent(net=net, sampled_points=sampling)
    assert jnp.allclose(grad(lambda theta: sc.differentiable_topological_loss(theta))(theta), 4.86983)      #TODO : trhe same, with
                                                                                                            # severall values of theta,
                                                                                                            # and explicit formula