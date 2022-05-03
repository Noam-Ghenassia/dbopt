import pytest
import jax.numpy as jnp
from jax import grad, jacrev
from dbopt.DB_Top_opt import SingleCycleDecisionBoundary
from dbopt.Bumps import Bumps
from dbopt.DB_sampler import DecisionBoundarySampler

__author__ = "Noam Ghenassia"
__copyright__ = "Noam Ghenassia"
__license__ = "MIT"

def topological_loss_of_unit_square():
    pass