# %%
import pwd
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#%%
# importing relevant packages
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from dbopt.DB_sampler import DecisionBoundarySampler
from dbopt.Bumps import Bumps
from dbopt.Datasets import Spiral
from dbopt.FCNN import FCNN
from dbopt.DB_Top_opt import DecisionBoundrayOptimizer

# %%
seed = 23
key = random.PRNGKey(seed)


#%%
# testing the bumps function

bumps = Bumps()
fig, ax = plt.subplots()
bumps.plot(ax, contourf=True)

#%%
# sampling the 0-level set of the bumps function

bumps = Bumps()
sampler = DecisionBoundarySampler()

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
bumps.plot(ax1)
pts = sampler.get_points()
ax1.scatter(pts[:, 0], pts[:, 1], color='blue')

sampler.sample(theta=0, net=bumps.level)

pts = sampler.get_points()
bumps.plot(ax2)
ax2.scatter(pts[:, 0], pts[:, 1], color='red')


# %%
# testing the creation of the spiral dataset
key, ds_key = random.split(key)
spiral = Spiral(55, ds_key)
fig, ax = plt.subplots()
spiral.plot(ax)


#%%
# fitting the sprial dataset with the neural network of the FCNN class

network = FCNN(num_neurons_per_layer=[10, 10, 10, 10, 10, 2])
key, init_x_key = random.split(key)
x_init = random.uniform(init_x_key, (2,))
key, init_key = random.split(key)
params = network.init(init_key, x_init)

dataset = spiral.get_dataset()

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
spiral.plot(ax1)
network.plot_decision_boundary(params, ax1)

key, train_key = random.split(key)
params = network.train(train_key, params, dataset, 150)

spiral.plot(ax2)
network.plot_decision_boundary(params, ax2)

# %%
# sampling the decision boundary of the neural network
sampler = DecisionBoundarySampler(min=-10, max=10)
sampler.sample(params, network)
pts = sampler.get_points()
fig, ax = plt.subplots()
network.plot_decision_boundary(params, ax)
ax.scatter(pts[:, 0], pts[:, 1], color='red')


# %%
# optimizing the theta parameter of the bumps function to get a single
# feature in H1 of the 0-level set

bumps = Bumps()
net = lambda x, theta: bumps.level(x, theta)
theta = jnp.array(0.)

db_opt = DecisionBoundrayOptimizer(net, theta, 200, sampling_epochs=1000,
                                  update_epochs=25, optimization_lr=0.01)

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
bumps.plot(ax1)
pts = db_opt.get_points()
ax1.scatter(pts[:, 0], pts[:, 1], color='red')

theta = db_opt.optimize(n_epochs=15)

bumps.plot(ax2, theta)
pts = db_opt.get_points()
ax2.scatter(pts[:, 0], pts[:, 1], color='red')

# %%
