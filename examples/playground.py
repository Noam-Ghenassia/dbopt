#%%
import numpy as np
import jax.numpy as jnp
import jax.random as random

#%%
a = np.array([0, 1, 2])
b = jnp.array([0, 1, 2])

print(np.argwhere(a < .2))
print(jnp.argwhere(b < .2))
# %%
c = jnp.array([[0, 1, 2],[3, 4, 5]])
print(a.ndim)
# %%
print(jnp.abs(b))
# %%
print(jnp.argmax(c, axis=1))
# %%
print(jnp.exp(b))
# %%

s = 5
a = random.uniform(random.PRNGKey(0), shape=(s,))
# %%

# %%

# %%
