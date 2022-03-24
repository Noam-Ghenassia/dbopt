import pytest
from dbopt.DB_sampler import DB_sampler
import jax.numpy as jnp

__author__ = "Noam Ghenassia"
__copyright__ = "Noam Ghenassia"
__license__ = "MIT"


def test_random_sampling():
    sampler = DB_sampler(n_points=30, input_dim=3, min=-2, max=1)
    assert sampler.points.ndim == 2
    assert sampler.points.shape[0] == 30
    assert sampler.points.shape[1] == 3
    assert jnp.min(sampler.points[:, 0]) >= -2
    assert jnp.max(sampler.points[:, 2]) <= 1


#def test_main(capsys):
#    """CLI Tests"""
#    # capsys is a pytest fixture that allows asserts agains stdout/stderr
#    # https://docs.pytest.org/en/stable/capture.html
#    main(["7"])
#    captured = capsys.readouterr()
#    assert "The 7-th Fibonacci number is 13" in captured.out