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
from dbopt.DB_Top_opt import DecisionBoundrayGradient

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
#net = lambda x, r: (x ** 2).sum() - r ** 2
r = jnp.array([3.0])
#x = jnp.array([[1, 1], [2, 0]])
#x_normalized = x / jnp.sqrt((x**2).sum(axis=1)).reshape(-1, 1)  # lies on the unit circle


y = jnp.array([[1, 1], [-1, 1], [2, 0]])
#y = jnp.array([[1, 1]])
y_normalized = y / jnp.sqrt((y**2).sum(axis=1)).reshape(-1, 1)
y_normalized = 3*y_normalized

#net = lambda x, r: (x ** 2).sum(axis=1) - r ** 2 * jnp.ones(x.shape[0])
net = lambda x, r: (x ** 2).sum(axis=1) - r **2 * jnp.ones(x.shape[0])
db_opt = DecisionBoundrayGradient(net, y_normalized)
print(y_normalized)
print((y_normalized ** 2).sum(axis=1))
print(r ** 2 * jnp.ones(y_normalized.shape[0]))
print(net(y_normalized, r))

# %%
from jax import grad, jacrev
#print(grad(lambda r: db_opt.t_star(r).sum())(r))
print(jacrev(lambda r: db_opt.t_star(r))(r))

# %%
opt_t = jnp.zeros(3)
#print((db_opt._parametrization_normal_lines(y_normalized, r) **2).sum(axis=1))
print((db_opt._parametrization_normal_lines(opt_t, r) **2).sum(axis=1))
print(r ** 2 * jnp.ones(3))
print((db_opt._parametrization_normal_lines(opt_t, r) **2).sum(axis=1) - r ** 2 * jnp.ones(3))



#return (self._parametrization_normal_lines(t, theta) ** 2).sum(axis=1)\
#    - theta ** 2 * jnp.ones(self.sampled_points.shape[0])



# %%
print(net(y_normalized, r))
# %%
