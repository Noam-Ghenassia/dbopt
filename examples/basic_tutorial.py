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
from dbopt.Bumps import Bumps
bumps = Bumps()
fig, ax = plt.subplots()
bumps.plot(ax)

#%%
from dbopt.DB_sampler import DB_sampler
from dbopt.Bumps import Bumps
from jax import value_and_grad
from jax.experimental.optimizers import adam
bumps = Bumps()
sampler = DB_sampler()
sampler.sample(bumps.level)



"""loss = sampler._loss(bumps.level)
print(loss)
opt_init, opt_update, get_points = adam(1e-2)
opt_state = opt_init(sampler.points)
value, grads = value_and_grad(DB_sampler._loss)(get_points(opt_state), bumps.level)"""








# %%
from dbopt.Datasets import Spiral
spiral = Spiral(100)
fig, ax = plt.subplots()
spiral.plot(ax)


#%%
from dbopt.FCNN import FCNN
from dbopt.Datasets import Spiral
nn = FCNN()
spiral = Spiral(100)
data, labels = spiral.get_dataset()
nn.train(data, labels)


# %%
