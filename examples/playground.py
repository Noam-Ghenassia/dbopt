#%%
from dbopt.ImpliciteDifferentiationSample import ImpliciteDifferentationSample
import numpy as np
import jax.numpy as jnp
import jax.random as random

#%%
from jax import grad
import jax.numpy as jnp
from jaxopt.implicit_diff import custom_root

t = jnp.array([1., 2., 3.])
ids = ImpliciteDifferentationSample()

sum = lambda a: jnp.sum(a)

grads = grad(lambda t: sum(ids.x_star(t)))(t)


# %%

