# %%
import pwd
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


#%%
import numpy as np
import jax.numpy as jnp
import jax.random as random

from dbopt.DB_sampler import DB_sampler
from dbopt.Bumps import Bumps
from dbopt.Datasets import Spiral
from dbopt.FCNN import FCNN
from dbopt.DB_Top_opt import DB_Top_opt

#%%
# bumps = Bumps()
# opt = DB_Top_opt(bumps.level, n_sampling=20)
# print(opt.toploss(theta=0))   #same error



# # %%
# from dbopt.ImpliciteDifferentiationSample import ImpliciteDifferentationSample as d1
# #from dbopt.ImpliciteDifferentiationSample import DifferentiationSample as d2

# from jax import grad
# import jax.numpy as jnp
# #c2=d2()
# #print(grad(c2.f)(2.))
# #print(jnp.exp(2.))

# c1=d1()
# a = jnp.array([2.0])
# c1.implicite_differentiation(a)
# # %%
# import jax.numpy as jnp

# from jax import grad
# from dbopt.ImpliciteDifferentiationSample import ImpliciteDifferentationSample

# # %%
# d1 = ImpliciteDifferentationSample()

# grad(d1.implicite_differentiation)(jnp.array([1.0]))
# # %%
# l = [[1, 1], [2, 2], [3, 3], [4, 4]]
# print([i for i in l if i[1]%2==0])
# #print(l[l[:, 1]%2==0])

# # %%

# bumps = Bumps()
# opt = DB_Top_opt(bumps.level, n_sampling=1000)
# points = opt.get_points()

# %%

#net = lambda x, r: (x ** 2).sum() - r ** 2 * x.shape[0]
net = lambda x, r: (x ** 2).sum() - r ** 2

x = jnp.array([[1, 1],
               [2, 0]])

x_normalized = x / jnp.sqrt((x**2).sum(axis=1)).reshape(-1, 1)  # lies on the unit circle

y = jnp.array([[1, 1], [-1, 1], [2, 0]])
y_normalized = y / jnp.sqrt((y**2).sum(axis=1)).reshape(-1, 1)

#db_opt = DB_Top_opt(net, n_sampling=0)
db_opt = DB_Top_opt(net, x_normalized)
# %%
from jax import grad
#db_opt.sampled_points = x_normalized
#db_opt.n_sampling = 2

r = jnp.array([1.0])

#grad(lambda r: ((x_normalized * r) ** 2).sum())(r)

# t_star is the function that gives the parameters t such that the new points
# lie on the circle of radius r. there are 2 points, and changing the value
# of r (making it bigger) changes the error of each point by a factor 1.
print("tstar : ", db_opt.t_star(r))
grad(lambda r: db_opt.t_star(r).sum())(r)

# %%
db_opt.sampled_points
# %%
db_opt.t_star(r).sum()





# %%
x = jnp.array([[1, 1],
               [2, 0]])
print((x**2).sum(axis=1))
# %%
print(jnp.sqrt((x**2).sum(axis=1)))
# %%
print(jnp.sqrt((x**2).sum(axis=1)).reshape(-1, 1))
# %%
print(x_normalized)
# %%
print(x / jnp.sqrt((x**2).sum(axis=1)))
# %%
print(x / jnp.sqrt((x**2).sum(axis=1)).reshape(-1, 1))
# %%
a = jnp.array([[ 0.7071068,0.7071068],[-0.7071068,0.7071068],[ 1.,0.]])
b = jnp.zeros([3, 1])
print(a + b*a)
# %%
