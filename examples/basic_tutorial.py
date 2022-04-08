# %%
import pwd
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#%%
#importing relevant packages
from jax import random
import matplotlib.pyplot as plt

from dbopt.DB_sampler import DB_sampler
from dbopt.Bumps import Bumps
from dbopt.Datasets import Spiral
from dbopt.FCNN import FCNN


#%%
#testing the bumps function

bumps = Bumps()
fig, ax = plt.subplots()
bumps.plot(ax)

#%%
#sampling the 0-level set of the bumps function

bumps = Bumps()
sampler = DB_sampler()

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
bumps.plot(ax1)
pts = sampler.get_points()
ax1.scatter(pts[:, 0], pts[:, 1], color='blue')

sampler.sample(bumps.level)

pts = sampler.get_points()
bumps.plot(ax2)
ax2.scatter(pts[:, 0], pts[:, 1], color='red')


# %%
#testing the creation of the spiral dataset

spiral = Spiral(100)
fig, ax = plt.subplots()
spiral.plot(ax)


#%%
#fitting the sprial dataset with the neural network of the FCNN class

nn = FCNN(num_epochs=130)
spiral = Spiral(100)
data, labels = spiral.get_dataset()

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
spiral.plot(ax1)
nn.plot_decision_boundary(ax1)

nn.train(data, labels)

nn.plot_decision_boundary(ax2)
spiral.plot(ax2)

# %%
fig, ax = plt.subplots()
nn.plot_decision_boundary(ax)
spiral.plot(ax)

# %%
