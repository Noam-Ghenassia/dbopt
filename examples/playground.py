# %%
import pwd
from IPython import get_ipython

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


#%%
import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax import grad
import matplotlib.pyplot as plt

from dbopt.DB_sampler import DecisionBoundarySampler
from dbopt.Bumps import Bumps
from dbopt.Datasets import Spiral
from dbopt.FCNN import FCNN
#from dbopt.DB_Top_opt import DecisionBoundrayGradient, TopologicalLosses
from dbopt.DB_Top_opt import DecisionBoundrayGradient
from dbopt.DB_Top_opt import SingleCycleDecisionBoundary
from dbopt.DB_Top_opt import SingleCycleAndConnectedComponent
from dbopt.DB_Top_opt import DecisionBoundrayOptimizer
from dbopt.persistent_gradient import PersistentGradient  # type: ignore

# %%
#bumps = Bumps()
#fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
#bumps.plot(ax1, True)
#bumps.plot(ax2, False)

# %%
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

# %%
# d1 = ImpliciteDifferentationSample()

# grad(d1.implicite_differentiation)(jnp.array([1.0]))
# # %%
# l = [[1, 1], [2, 2], [3, 3], [4, 4]]
# print([i for i in l if i[1]%2==0])
# #print(l[l[:, 1]%2==0])

# %%

# bumps = Bumps()
# opt = DB_Top_opt(bumps.level, n_sampling=1000)
# points = opt.get_points()

# %%

#r = jnp.array([3.0])

#x = jnp.array([[1, 1], [2, 0]])
#x_normalized = 3*(x / jnp.sqrt((x**2).sum(axis=1)).reshape(-1, 1))

#y = jnp.array([[1, 1], [-1, 1], [1, 2], [2, 1], [0, -1]])
#y_normalized = y / jnp.sqrt((y**2).sum(axis=1)).reshape(-1, 1)
#y_normalized = 3*y_normalized

#net = lambda x, r: (x ** 2).sum(axis=1) - r **2 * jnp.ones(x.shape[0])
#db_opt = DecisionBoundrayGradient(net, y_normalized)
#print(y_normalized)
#print((y_normalized ** 2).sum(axis=1))
#print(r ** 2 * jnp.ones(y_normalized.shape[0]))
#print(net(y_normalized, r))

# %%
#from jax import grad, jacrev
#print("grad : ", grad(lambda r: db_opt.t_star(r).sum())(r))        #should give [n_points]
#print("jacobian : ", jacrev(lambda r: db_opt.t_star(r))(r))           # should give [[1.], [1.], ..., [1]] n_points times



# %%
#net = lambda x, r: x.sum(axis=1)-r*jnp.ones(x.shape[0])
#r = jnp.array([3.])
#x = jnp.array([[1., 2.], [2., 1.], [4., -1.], [6., -3.]])
#db_opt = DecisionBoundrayGradient(net, x)

# %%
#from jax import grad, jacrev
#print("grad : ", grad(lambda r: db_opt.t_star(r).sum())(r))
#print("jacobian : ", jacrev(lambda r: db_opt.t_star(r))(r))

# %%
#net = lambda x, r: (x ** 2).sum(axis=1) - r **2 * jnp.ones(x.shape[0])
#x = jnp.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
#r = jnp.array([jnp.sqrt(2)])
#sc = SingleCycleDecisionBoundary(net, x)
#print(sc.topological_loss_with_gradient(r))
#print(grad(lambda r: sc.topological_loss_with_gradient(r))(r))
# %%
#bumps = Bumps()
#theta = jnp.array([0.])
#net = lambda x, theta: bumps.level(x, theta)
#sampling = jnp.array([[-0.5325, 0.], [0.5325, 0]])
#sc = SingleCycleAndConnectedComponent(net=net, sampled_points=sampling)
#print(grad(lambda theta: sc.topological_loss_with_gradient(theta))(theta))


# %%
#bumps = Bumps()
#sampler = DecisionBoundarySampler(n_points=1000)
#net = lambda x, theta: bumps.level(x, theta)
# FIRST VALUE OF THETA : 0.
#theta1 = 0.
#sampling = sampler.sample(theta=theta1, net=net)
#sc = SingleCycleAndConnectedComponent(net=net, sampled_points=sampling)
#print(grad(lambda theta: sc.topological_loss_with_gradient(theta))(theta1))
#print(sc.topological_loss_with_gradient(theta1))
# SECOND VALUE OF THETA : 0.2
#theta2 = 0.2
#sampling = sampler.sample(theta=theta2, net=net)
#sc = SingleCycleAndConnectedComponent(net=net, sampled_points=sampling)
#print(grad(lambda theta: sc.topological_loss_with_gradient(theta))(theta2))
#print(sc.topological_loss_with_gradient(theta2))

# %%
#bumps = Bumps()
#net = lambda x, theta: bumps.level(x, theta)
#theta = jnp.array(0.)

#db_opt = DecisionBoundrayOptimizer(net, theta, 200, sampling_epochs=1000,
#                                   update_epochs=3, optimization_lr=0.01)

#fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
#bumps.plot(ax1)
#pts = db_opt.get_points()
#ax1.scatter(pts[:, 0], pts[:, 1], color='red')

#theta = db_opt.optimize(n_epochs=15)

#bumps.plot(ax2, theta)
#pts = db_opt.get_points()
#ax2.scatter(pts[:, 0], pts[:, 1], color='red')


# %%
nn = FCNN()
# HOW TO GET THE PARAMETERS ?