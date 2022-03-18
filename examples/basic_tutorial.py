#%%
from jax import random
import matplotlib.pyplot as plt


#%%
from dbopt.DB_sampler import DB_sampler
key = random.PRNGKey(1)
sampler = DB_sampler()

pts = sampler._random_sampling()
print(pts.shape)
print(type(pts))
fig, ax = plt.subplots()
ax.scatter(pts[:, 0], pts[:, 1])


#%%
from dbopt.FCNN import FCNN
NN = FCNN()

#%%
from dbopt.Bumps import Bumps
bumps = Bumps()
fig, ax = plt.subplots()
bumps.plot(ax)
# %%
from dbopt.Spiral import Spiral
spiral = Spiral(500)
fig, ax = plt.subplots()
spiral.plot(ax)
# %%
