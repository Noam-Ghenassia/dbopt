#%%
import numpy as np
import jax.numpy as jnp
import jax.random as random

#%%
a = jnp.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
print(a)
b = jnp.linalg.norm(a, axis=1)
print(b)
print(jnp.divide(a, b[:, None]))


# %%
a = 2.
b = jnp.array(a)
print(b)
print(type(b))
# %%
