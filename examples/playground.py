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

from src.dbopt.DB_sampler_2d import DecisionBoundarySampler
from src.dbopt.Bumps import Bumps
from src.dbopt.Datasets import Spiral
from src.dbopt.FCNN import FCNN
from src.dbopt.DB_Top_opt_2d import DecisionBoundrayGradient
from src.dbopt.DB_Top_opt_2d import DecisionBoundrayOptimizer
from src.dbopt.persistent_gradient import PersistentGradient  # type: ignore
from src.dbopt.persistent_gradient import plot_persistence_diagram
#from dbopt.FCNN import _FCNN

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
from src.dbopt.Bumps import Bumps
bumps = Bumps()
net = bumps
theta = jnp.array(0.)

db_opt = DecisionBoundrayOptimizer(net, theta, 200, sampling_epochs=1000,
                                  update_epochs=15, optimization_lr=0.03,
                                  loss_name="single_cycle_and_connected_component",
                                  with_logits=False, with_dataset=False,
                                  min_x=-1.5, max_x=1.5, min_y=-0.6, max_y=0.6)

print('n points : ', db_opt.get_points().shape[0])

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
bumps.plot(ax1)
pts = db_opt.get_points()
ax1.scatter(pts[:, 0], pts[:, 1], color='red')

theta = db_opt.optimize(n_epochs=13)

bumps.plot(ax2, theta)
pts = db_opt.get_points()
ax2.scatter(pts[:, 0], pts[:, 1], color='red')
#plt.savefig('bumps_optimization.pdf')

#########################################################################
######################## DONE WITH THE BUMPS ############################
#########################################################################



# # %%
# #initializing dataset
# seed = 24
# key = random.PRNGKey(seed)

# key, ds_key = random.split(key)
# spiral = Spiral(75, ds_key)
# dataset = spiral.get_dataset()

# #fig, ax = plt.subplots()
# #spiral.plot(ax)
# #plt.savefig('spiral_dataset.pdf')

# #%%
# # fitting the network

# net = FCNN(num_neurons_per_layer=[10, 10, 10, 10, 10, 2])
# key, init_x_key = random.split(key)
# x_init = random.uniform(init_x_key, (2,))
# key, init_key = random.split(key)
# params = net.init(init_key, x_init)

# key, train_key = random.split(key)
# params = net.train(train_key, params, dataset, 150)

# print('done fitting the network')
# fig, ax = plt.subplots(1, figsize=(11, 11))
# spiral.plot(ax)
# net.plot_decision_boundary(params, ax, x_min=-15., x_max=15., y_min=-10., y_max=15.)
# #plt.savefig('DB.pdf')
# #%%
# #initializing db_opt
# fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
# spiral.plot(ax1)
# net.plot_decision_boundary(params, ax1, x_min=-15., x_max=15., y_min=-10., y_max=15.)

# db_opt = DecisionBoundrayOptimizer(net, params, n_sampling=1600, sampling_epochs=500,
#                                   loss_name="single_connected_component",
#                                   update_epochs=15, optimization_lr=2e-2, min_x=-14., max_x=10., min_y=-7., max_y=12.5)
# print('done sampling')          #lr = 1e-2, max_y = 12.5, update_epochs = 10

# spiral.plot(ax2)
# net.plot_decision_boundary(params, ax2, x_min=-15., x_max=15., y_min=-10., y_max=15.)
# ax2.scatter(db_opt.get_points()[:, 0], db_opt.get_points()[:, 1], color='green')

# print('points shape : ', db_opt.get_points().shape)

# # %%
# #PD plots before
# fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(30, 11.25))

# spiral.plot(ax1)
# net.plot_decision_boundary(params, ax1, x_min=-15., x_max=15., y_min=-10., y_max=15.)
# ax1.scatter(db_opt.get_points()[:, 0], db_opt.get_points()[:, 1], color='green')

# pg = PersistentGradient()
# dbgrad = DecisionBoundrayGradient(net, db_opt.get_points())
# points = db_opt.get_points()
# normal_vectors = dbgrad._normal_unit_vectors(params)
# diag1 = pg._computing_persistence_with_gph(points, jnp.zeros_like(points))
# diag2 = pg._computing_persistence_with_gph(points, normal_vectors)
# plot_persistence_diagram(diag1, ax2)
# plot_persistence_diagram(diag2, ax3)
# #plt.savefig('PD_before.PDF')

# #%%
# #running db_opt
# params = db_opt.optimize(n_epochs=8, dataset=dataset)

# fig, ax = plt.subplots(1, figsize=(11, 11))
# spiral.plot(ax)
# net.plot_decision_boundary(params, ax, x_min=-15., x_max=15., y_min=-10., y_max=15.)

# #%%
# #saving DB after
# fig, ax = plt.subplots(1, figsize=(11, 11))
# spiral.plot(ax)
# net.plot_decision_boundary(params, ax, x_min=-15., x_max=15., y_min=-10., y_max=15.)
# plt.savefig('DB_after.pdf')

# #%%
# # displaying PD after
# fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(30, 11.25))

# spiral.plot(ax1)
# net.plot_decision_boundary(params, ax1, x_min=-15., x_max=15., y_min=-10., y_max=15.)
# ax1.scatter(db_opt.get_points()[:, 0], db_opt.get_points()[:, 1], color='green')

# pg = PersistentGradient()
# dbgrad = DecisionBoundrayGradient(net, db_opt.get_points())
# points = db_opt.get_points()
# normal_vectors = normal_vectors = dbgrad._normal_unit_vectors(params)
# diag3 = pg._computing_persistence_with_gph(points, jnp.zeros_like(points))
# diag4 = pg._computing_persistence_with_gph(points, normal_vectors)
# plot_persistence_diagram(diag3, ax2)
# plot_persistence_diagram(diag4, ax3)
# plt.savefig('PD_after.pdf')

############################################################################
############################## SPARSE EXAMPLE ##############################
############################################################################

# %%
#initializing dataset
seed = 23
key = random.PRNGKey(seed)

key, ds_key = random.split(key)
spiral = Spiral(30, ds_key)
dataset = spiral.get_dataset()

fig, ax = plt.subplots()
spiral.plot(ax)
#plt.savefig('spiral_dataset.pdf')

#%%
# fitting the network

net = FCNN(num_neurons_per_layer=[10, 10, 10, 10, 10, 2])
key, init_x_key = random.split(key)
x_init = random.uniform(init_x_key, (2,))
key, init_key = random.split(key)
params = net.init(init_key, x_init)

key, train_key = random.split(key)
params = net.train(train_key, params, dataset, 150)

print('done fitting the network')
fig, ax = plt.subplots(1, figsize=(11, 11))
spiral.plot(ax)
net.plot_decision_boundary(params, ax, x_min=-15., x_max=15., y_min=-10., y_max=15.)
#plt.savefig('DB.pdf')
#%%
#initializing db_opt
fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 11))
spiral.plot(ax1)
net.plot_decision_boundary(params, ax1, x_min=-15., x_max=15., y_min=-10., y_max=15.)

db_opt = DecisionBoundrayOptimizer(net, params, n_sampling=1600, sampling_epochs=500,
                                  loss_name="single_connected_component",
                                  update_epochs=15, optimization_lr=2e-2, min_x=-14., max_x=10., min_y=-7., max_y=12.5)
print('done sampling')          #lr = 1e-2, max_y = 12.5, update_epochs = 10

spiral.plot(ax2)
net.plot_decision_boundary(params, ax2, x_min=-15., x_max=15., y_min=-10., y_max=15.)
ax2.scatter(db_opt.get_points()[:, 0], db_opt.get_points()[:, 1], color='green')

print('points shape : ', db_opt.get_points().shape)

# %%
#PD plots before
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(30, 11.25))

spiral.plot(ax1)
net.plot_decision_boundary(params, ax1, x_min=-15., x_max=15., y_min=-10., y_max=15.)
ax1.scatter(db_opt.get_points()[:, 0], db_opt.get_points()[:, 1], color='green')

pg = PersistentGradient()
dbgrad = DecisionBoundrayGradient(net, db_opt.get_points())
points = db_opt.get_points()
normal_vectors = dbgrad._normal_unit_vectors(params)
diag1 = pg._computing_persistence_with_gph(points, jnp.zeros_like(points))
diag2 = pg._computing_persistence_with_gph(points, normal_vectors)
plot_persistence_diagram(diag1, ax2)
plot_persistence_diagram(diag2, ax3)
#plt.savefig('PD_before.PDF')

#%%
#running db_opt
params = db_opt.optimize(n_epochs=1, dataset=dataset)

fig, ax = plt.subplots(1, figsize=(11, 11))
spiral.plot(ax)
net.plot_decision_boundary(params, ax, x_min=-15., x_max=15., y_min=-10., y_max=15.)

#%%
#saving DB after
fig, ax = plt.subplots(1, figsize=(11, 11))
spiral.plot(ax)
net.plot_decision_boundary(params, ax, x_min=-15., x_max=15., y_min=-10., y_max=15.)
#plt.savefig('DB_after.pdf')

#%%
# displaying PD after
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(30, 11.25))

spiral.plot(ax1)
net.plot_decision_boundary(params, ax1, x_min=-15., x_max=15., y_min=-10., y_max=15.)
ax1.scatter(db_opt.get_points()[:, 0], db_opt.get_points()[:, 1], color='green')

pg = PersistentGradient()
dbgrad = DecisionBoundrayGradient(net, db_opt.get_points())
points = db_opt.get_points()
normal_vectors = normal_vectors = dbgrad._normal_unit_vectors(params)
diag3 = pg._computing_persistence_with_gph(points, jnp.zeros_like(points))
diag4 = pg._computing_persistence_with_gph(points, normal_vectors)
plot_persistence_diagram(diag3, ax2)
plot_persistence_diagram(diag4, ax3)
#plt.savefig('PD_after.pdf')
# %%
