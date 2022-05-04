from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Dict, Callable
from functools import partial

from jax import numpy as jnp
from jax import jit, jacfwd, grad
from jaxopt.implicit_diff import custom_root
import numpy as np
import optax


from dbopt.DB_sampler import DecisionBoundarySampler
from dbopt.persistent_gradient import PersistentGradient


class DecisionBoundrayGradient():
    """This class allows to modify the parameters of a function (typically, the
    weights of a neural network) in order to optimize the homology of the
    simplicial complex constructed on a set of points sampled from the 0-level
    set of that function (or the decision boundary of the neural network). The
    optimization minimizes a user provided function of the persistence diagram of
    the filtration of the point cloud.
    
    Note : a future improvement would be to add the ability for the user to only
    provide a set of Betti numbers instead of the actual function to optimize.
    """
    
    def __init__(self, net: callable, sampled_points: jnp.array):
        self.net: Callable[[jnp.array], jnp.array] = net
        self.sampled_points = sampled_points
    
    def get_points(self):
        return self.sampled_points
    
    def _normal_unit_vectors(self, theta: Union[jnp.array, Dict[str, jnp.array]]):
        """This function returns a set of vectors that are normal to the decision boundary
        at the points that were sampled from it by the sampler.

        Args:
            net (function): the network that is being optimized.

        Returns:
            jnp.array: an n*d matrix with rows the normal vectors of the decision boundary
            evaluated at the points sampled by the sampler.
        """
        normal_vectors = grad(lambda x : self.net(x, theta).sum())(self.sampled_points)
        norms = jnp.linalg.norm(normal_vectors, axis=1).reshape(-1, 1)
        return normal_vectors / norms
    
    def _parametrization_normal_lines(self, t: jnp.array, theta: jnp.array):
        """This function gives a new set of points that depend on the points initially sampled
        by the sampler, but that only depend on a single real coordinate each. These new
        points are bound to move along a line that is normal to the decision boundary
        and passes through one of the original points (each). When t=0, the returned points
        equal the old ones.

        Args:
            t (jnp.array): the parameters that define the position of the new points.
            net (function): the network that is being optimized

        Returns:
            jnp.array: the new points.
        """
        
        return self.sampled_points + jnp.expand_dims(t, 1) * self._normal_unit_vectors(theta)


    def _optimality_condition(self, t, theta):
        """This function is the optimality condition that implicitly defines
        points*(theta) : for a given theta, the optimality condition is zero
        when the points lie on the decision boundary.

        Args:
            t (jnp.array): the coordinates of the new points along the normal vectors. In
            the setting of jaxopt, these are the variables optimized in the inner
            optimization problem.
            theta (jnp.array): the parameters of the network.
            net (function): the function parametrized by theta.

        Returns:
            float: the points loss, i.e., the sum of the squared distances to the DB
        """
        
        # Working solution for special case
        #print("new points : ", self._parametrization_normal_lines(t, theta).shape)
        #return (self._parametrization_normal_lines(t, theta) ** 2).sum(axis=1)\
        #    - theta ** 2 * jnp.ones(self.sampled_points.shape[0])
        
        points_along_normal_lines = self._parametrization_normal_lines(t, theta)
        logits = self.net(points_along_normal_lines, theta)
        
        if logits.ndim == 2:
            #deviation_from_decision_boundary = (logits[:, 0]-logits[:, 1])**2
            deviation_from_decision_boundary = logits[:, 0]-logits[:, 1]
        else :
            #deviation_from_decision_boundary = logits**2
            deviation_from_decision_boundary = logits
        return deviation_from_decision_boundary
    
    
    def _inner_problem(self, t_init, theta):
        """This function is the inner optimization problem. It simply samples (with the new
        points) the decision boundary of the network, but with the custom root decorator it
        is possible to get the jacobian of the optimal points wrt the parameters
        of net (theta), which will be necessary in the chain rule that allows to differentiate
        the topological loss wrt theta.

        Args:
            t (jnp.array): the parameters that define the position of the new points
            theta (jnp.array): the points that sample the decision boundary
            net (function): the function parametrized by theta
        """

        del t_init
        return jnp.zeros(self.sampled_points.shape[0])
    
        
    def t_star(self, theta):
        """ This function returns the optimal value of t, i.e., the value of t that
        minimizes the distance between the points and the decision boundary.
        """
        t_init = None
        return custom_root(self._optimality_condition)\
            (self._inner_problem)(t_init, theta)
    
######################################################################


class TopologicalLoss(ABC):

    def __init__(self, net, sampled_points):
        self.sampled_points = sampled_points
        self.persistent_gradient = PersistentGradient()
        self.db_grad = DecisionBoundrayGradient(net, sampled_points)
    
    @abstractmethod
    def _toploss(self):
        pass
    
    def update_sampled_points(self, new_points):
        self.sampled_points = new_points
        
    
    def topological_loss_with_gradient(self, theta):
        
        t = self.db_grad.t_star(theta)
        parametrized_sampling = self.db_grad._parametrization_normal_lines(t, theta)
        return self._toploss(parametrized_sampling)
    

class SingleCycleDecisionBoundary(TopologicalLoss):
    
    def __init__(self, net, sampled_points):
        super().__init__(net, sampled_points)
    
    def _toploss(self, parametrized_sampling):
        
        pers_diag = self.persistent_gradient._computing_persistence_with_gph(parametrized_sampling)
        # select only the pairs that correspond to 1D features
        #H1 = pers_diag[pers_diag[:, 2]==1]     # is there a way to make this parallelizable ?
        H1 = jnp.array([jnp.asarray(pers_pair) for pers_pair in pers_diag if pers_pair[2]==1])
        lifetimes = H1[:, 1] - H1[:, 0]
        largest = jnp.argmax(lifetimes)
        largest_cycle = lifetimes[largest]
        other_cycles = jnp.delete(lifetimes, largest)
        return jnp.sum(other_cycles**2) - largest_cycle**2

class SingleCycleAndConnectedComponent(TopologicalLoss):
    
    def __init__(self, net, sampled_points):
        super().__init__(net, sampled_points)
    
    def _toploss(self, parametrized_sampling):
        pers_diag = self.persistent_gradient._computing_persistence_with_gph(parametrized_sampling)
        H0 = jnp.array([jnp.asarray(pers_pair) for pers_pair in pers_diag if pers_pair[2]==0])
        H1 = jnp.array([jnp.asarray(pers_pair) for pers_pair in pers_diag if pers_pair[2]==1])
        cycles_lifetimes = H1[:, 1] - H1[:, 0]
        largest = jnp.argmax(cycles_lifetimes)
        largest_cycle = cycles_lifetimes[largest]
        other_cycles = jnp.delete(cycles_lifetimes, largest)
        last_merge = jnp.max(H0[:, 1])
        return jnp.sum(other_cycles**2) - largest_cycle**2 + last_merge**2
    
    
class SingleConnectedComponent(TopologicalLoss):
    
    def __init__(self, net, sampled_points):
        super().__init__(net, sampled_points)
    
    def _toploss(self, parametrized_sampling):
        pers_diag = self.persistent_gradient._computing_persistence_with_gph(parametrized_sampling)
        H0 = jnp.array([jnp.asarray(pers_pair) for pers_pair in pers_diag if pers_pair[2]==0])
        last_merge = jnp.max(H0[:, 1])
        return last_merge**2
        

##################################################################

    

class DecisionBoundrayOptimizer():
    
    #def __init__(self, net, theta, n_sampling, toploss: TopologicalLoss,
    #             sampling_epochs=1000, update_epochs=3, sampling_lr=0.01, optimization_lr=0.01):
    def __init__(self, net, theta, n_sampling,
                 sampling_epochs=1000, update_epochs=3, sampling_lr=0.01, optimization_lr=0.01):
        self.net = net
        self.theta = theta
        self.update_epochs = update_epochs
        self.optimization_lr = optimization_lr
        self.sampler = DecisionBoundarySampler(n_points=n_sampling)
        self.sampled_points = self.sampler.sample(theta, net, points=None, lr=sampling_lr, epochs=sampling_epochs)
        self.toploss = SingleCycleDecisionBoundary(net=net, sampled_points=self.sampled_points)
        #self.toploss = toploss(net=net, sampled_points=self.sampled_points)
    
    def _update_sampled_points(self):
        """This function uodates the sampled points so they remain on the decsion boundary
        after it was updated. It should be called after each optimization step.
        """
        #print(self.sampled_points)
        new_points = self.sampler.sample(self.theta, self.net, self.sampled_points,
                                         epochs=self.update_epochs)
        self.sampled_points = new_points
        self.toploss.update_sampled_points(new_points)


    def optimize(self, n_epochs):

        theta = jnp.array(self.theta)
        optimizer = optax.adam(self.optimization_lr)
        #params = {'theta': theta}
        params = theta
        opt_state = optimizer.init(params)
        #opt_state = optimizer.init(theta)
        loss = lambda x: self.toploss.topological_loss_with_gradient(x)     #in later versions this should include cross entropy

        for epoch in range(n_epochs):
            #grads = grad(loss)(params['theta'])
            grads = grad(loss)(theta)
            print("grads : ", grads)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            #theta = optax.apply_updates(theta, updates)
            print("theta before : ", self.theta)
            self.theta = params
            print("theta after : ", self.theta)
            self._update_sampled_points()

            #if epoch % self.update_epochs==self.update_epochs-1:
            #    self._update_sampled_points()
        
        return self.theta
    
    def get_points(self):
        return self.sampled_points

                
                
                
                
                
                
                
                

#def toploss(self, theta):
"""This the topological loss that is optimized by the class. It depends on the
value of theta.

Args:
    theta (jnp.array): the parameter(s) of the funcrion of which we optimize the
    decision boundary's homology.

Returns:
    jnp.array: the value of the topological loss.
"""
"""t_init = jnp.zeros_like(self.sampled_points[:, 0])
new_points = custom_root(self._optimality_condition)\
    (self._inner_problem)(t_init, theta)
return self.pg.single_cycle(new_points)"""