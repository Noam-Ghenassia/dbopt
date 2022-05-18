from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Dict, Callable
from functools import partial
from haiku import value_and_grad

from jax import numpy as jnp
from jax import jit, jacfwd, grad
from jaxopt.implicit_diff import custom_root
import numpy as np
import optax


from dbopt.DB_sampler import DecisionBoundarySampler
from dbopt.persistent_gradient import PersistentGradient


class DecisionBoundrayGradient():
    """This class allows to compute the gradient of the points with respect to the
    weights of the neural network that parametrize the manifold sampled by the points.
    To do this, it uses the implicit differentiation method provided in
    https://arxiv.org/abs/2105.15183. Given a sampling of the decision boundary, it
    creates a set of 1D parameters t (one parameter per point) that allow the points
    to move normally to the decision boundary. The optimality condition is then defined
    on t : since the original points already lie on the decision boundary, the optimal
    values of t are 0. This implicitely defines a function t^{\star}(theta), where theta
    are the parameters of the neural network. Jaxopt then allows to get the gradient of
    t^{\star} with respect to theta, which will then be used to compute the gradient of
    the topological loss.
    
    """
    
    def __init__(self, net: callable, sampled_points: jnp.array, with_logits=True):
        self.net: Callable[[jnp.array], jnp.array] = net
        self.sampled_points = sampled_points
        self.with_logits = with_logits
    
    def get_points(self):
        return self.sampled_points
    
    def _difference_of_the_logits(self, x, theta):
        """This function returns the difference of the logits, evaluated at the points in X.
        The gradient of the sum of its outputs gives the normal vectors of the decision
        boundary, evaluated at the points in X.

        Args:
            x (jnp.array): The points at which the function is evaluated.
            theta (jax.FrozenDict): The parameters of the network.

        Returns:
            _type_: _description_
        """
        logits = self.net.apply(theta, x)
        return logits[:, 1] - logits[:, 0]
    
    def _normal_unit_vectors(self, theta: Union[jnp.array, Dict[str, jnp.array]]):
        """This function returns a set of vectors that are normal to the decision boundary
        at the points that were sampled from it by the sampler.

        Args:
            theta (jnp.array): the weights of the neural network

        Returns:
            jnp.array: an n*d matrix with rows the normal vectors of the decision boundary
            evaluated at the points sampled by the sampler.
        """
        if self.with_logits:
            normal_vectors = grad(lambda x : self._difference_of_the_logits(x, theta).sum())(self.sampled_points)
        else:
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
            theta (jnp.array): the weights of the neural network

        Returns:
            jnp.array: the new points.
        """

        return self.sampled_points + jnp.expand_dims(t, 1) * self._normal_unit_vectors(theta)


    def _optimality_condition(self, t, theta):
        """This function is the optimality condition that implicitly defines
        t^{\star}(theta) : for a given theta, the optimality condition is zero
        when the points lie on the decision boundary.

        Args:
            t (jnp.array): the coordinates of the new points along the normal vectors. In
            the setting of jaxopt, these are the variables optimized in the inner
            optimization problem.
            theta (jnp.array): the parameters of the network.

        Returns:
            float: the points loss, i.e., the sum of the squared distances to the DB
        """
        points_along_normal_lines = self._parametrization_normal_lines(t, theta)
        #logits = self.net(points_along_normal_lines, theta)
        logits = self.net.apply(theta, points_along_normal_lines)
        
        if logits.ndim == 2:
            deviation_from_decision_boundary = logits[:, 0]-logits[:, 1]
        else :
            deviation_from_decision_boundary = logits
        return deviation_from_decision_boundary
    
    
    def _inner_problem(self, t_init, theta):
        """This function is the inner optimization problem. It returns the
        optimal t vectors that satisfies the optimality condition (namely,
        the 0 vector).

        Args:
            t_init (jnp.array): this parameter is not used, but is necessary for jaxopt.
            theta (jnp.array): the parameters of the network.
        """

        del t_init
        return jnp.zeros(self.sampled_points.shape[0])
    
        
    def t_star(self, theta):
        """ This function wraps the inner problem with the jaxopt function that allows
        to compute the gradient with respect to the parameters of the network.
        
        Args:
            theta (jnp.array): the parameters of the network.
        """
        t_init = None
        return custom_root(self._optimality_condition)\
            (self._inner_problem)(t_init, theta)
    
######################################################################


class TopologicalLoss(ABC):
    """This is the abstract from which the different topological losses are inherited.
    """
    
    def __init__(self, net, sampled_points, with_logits=True):
        """
        Args:
            net (Callable): the network that is being optimized.
            sampled_points (jnp.array): A sampling of the network's decision boundary.
        """
        self.sampled_points = sampled_points
        self.persistent_gradient = PersistentGradient()
        self.db_grad = DecisionBoundrayGradient(net, sampled_points, with_logits=with_logits)
    
    @abstractmethod
    def _toploss(self):
        pass
    
    def update_sampled_points(self, new_points):
        """This function allows to provide a new sampling of the decision boundary.
        It should be called each time the network's decision boundary changes, so
        the class can reliably keep track of its topology.

        Args:
            new_points (jnp.array): An updated sampling of the decision boundary.
        """
        self.sampled_points = new_points
        
    
    def differentiable_topological_loss(self, theta):
        """This function calls the specific topological loss indicated by the user,
        and computes its value at the sampled points, while allowing to backpropagate
        back to the weights of the network.

        Args:
            theta (jnp.array): The parameters of the network.

        Returns:
            jnp.array: The topological loss value.
        """
        t = self.db_grad.t_star(theta)
        parametrized_sampling = self.db_grad._parametrization_normal_lines(t, theta)
        return self._toploss(parametrized_sampling)
    

class SingleCycleDecisionBoundary(TopologicalLoss):
    """This class implements a topological loss that creates a single cycle in a 2D
    decision boundary.
    """
    
    def __init__(self, net, sampled_points):
        super().__init__(net, sampled_points)
    
    def _toploss(self, parametrized_sampling):
        
        pers_diag = self.persistent_gradient._computing_persistence_with_gph(parametrized_sampling)
        # select only the pairs that correspond to 1D features
        H1 = jnp.array([jnp.asarray(pers_pair) for pers_pair in pers_diag if pers_pair[2]==1])
        lifetimes = H1[:, 1] - H1[:, 0]
        largest = jnp.argmax(lifetimes)
        largest_cycle = lifetimes[largest]
        other_cycles = jnp.delete(lifetimes, largest)
        return jnp.sum(other_cycles**2) - largest_cycle**2


class SingleCycleAndConnectedComponent(TopologicalLoss):
    """This class implements a topological loss that creates a decision boundary with a 
    single cycle and a single connected component.
    """
    
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
    """This class implements a topological loss that creates a decision boundary with a
    single connected component.
    """
    
    def __init__(self, net, sampled_points):
        super().__init__(net, sampled_points)
    
    def _toploss(self, parametrized_sampling):
        pers_diag = self.persistent_gradient._computing_persistence_with_gph(parametrized_sampling)
        H0 = jnp.array([jnp.asarray(pers_pair) for pers_pair in pers_diag if pers_pair[2]==0])
        last_merge = jnp.max(H0[:, 1])
        return last_merge**2


def get_topological_loss(loss_name: str, net: Callable, sampled_points: jnp.array) -> TopologicalLoss:
    if "single_cycle_decision_boundary":
        return SingleCycleDecisionBoundary(net, sampled_points)
    elif "single_cycle_and_connected_component":
        return SingleCycleAndConnectedComponent(net, sampled_points)
    elif "single_connected_component":
        return SingleConnectedComponent(net, sampled_points)
    else:
        raise ValueError("The topological loss {} is not supported.".format(loss_name))
        

##################################################################

    

class DecisionBoundrayOptimizer():
    """This class allows to optimize the network's decision boundary, by
    performing gradient descent with the gradients of the topological loss.
    """
    
    
    def __init__(self, net, theta, n_sampling, loss_name: str, sampling_epochs=1000,
                 update_epochs=3, sampling_lr=0.01, optimization_lr=0.01, min=10., max=10.,
                 with_logits=True
                ):
        """_summary_

        Args:
            net (Callable): The network that is being optimized.
            theta (jnp.array): The parameters of the network.
            n_sampling (int): The number of points to sample the decision boundary.
            sampling_epochs (int, optional): The number of sampling epochs. Defaults to 1000.
            update_epochs (int, optional): The number of sampling epochs between optimization epochs. Defaults to 3.
            sampling_lr (float, optional): The learning rate of the sampler. Defaults to 0.01.
            optimization_lr (float, optional): The learning rate applied for the optimization. Defaults to 0.01.
        """
        self.net = net
        self.theta = theta
        self.update_epochs = update_epochs
        self.optimization_lr = optimization_lr
        self.sampler = DecisionBoundarySampler(n_points=n_sampling, min=min, max=max)
        self.sampled_points = self.sampler.sample(theta, net, points=None, lr=sampling_lr, epochs=sampling_epochs)
        self.toploss = get_topological_loss(loss_name, self.net, self.sampled_points, with_logits=with_logits)
    
    def _update_sampled_points(self):
        """This function updates the sampled points so they remain on the decsion boundary
        after it was updated. It should be called after each optimization step.
        """
        new_points = self.sampler.sample(self.theta, self.net, self.sampled_points,
                                         epochs=self.update_epochs)
        self.sampled_points = new_points
        self.toploss.update_sampled_points(new_points)


    def optimize(self, n_epochs):
        """This function allows to optimize the decision boundary. It uses the Adam
        optimizer to minimize the value of the topological loss. After each epoch,
        it updates the sampling of the decision boundary by calling the sample method
        of the DecisionBoundarySampler with the current points as initial points.

        Args:
            n_epochs (int): The number of optimization epochs.

        Returns:
            jnp.array: the parameters of the network after optimization.
        """
        # if isinstance(self.theta, float):
        #   theta = jnp.array(self.theta)
        # else :
        params = self.theta
        optimizer = optax.adam(self.optimization_lr)
        opt_state = optimizer.init(params)
        loss = lambda x: self.toploss.differentiable_topological_loss(x)     #in later versions this should include cross entropy

        for epoch in range(n_epochs):
            print('top epoch')
            #grads = grad(loss)(theta)
            grads = grad(loss)(self.theta)
            #loss, grads = value_and_grad(loss)(self.theta)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            self.theta = params
            self._update_sampled_points()
            print(loss(self.theta))
            print(grads)
        
        return self.theta
    
    def get_points(self):
        return self.sampled_points