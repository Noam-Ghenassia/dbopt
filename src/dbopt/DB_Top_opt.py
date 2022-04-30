from typing import Union, Dict, Callable
from functools import partial

from jax import numpy as jnp
from jax import jit, jacfwd, grad
from jaxopt.implicit_diff import custom_root
import numpy as np
import optax


from dbopt.DB_sampler import DB_sampler
from dbopt.persistent_gradient import PersistentGradient


class DB_Top_opt():         # rename DB_grad
    """This class allows to modify the parameters of a function (typically, the
    weights of a neural network) in order to optimize the homology of the
    simplicial complex constructed on a set of points sampled from the 0-level
    set of that function (or the decision boundary of the neural network). The
    optimization minimizes a user provided function of the persistence diagram of
    the filtration of the point cloud.
    
    Note : a future improvement would be to add the ability for the user to only
    provide a set of Betti numbers instead of the actual function to optimize.
    """
    
    #def __init__(self, net, n_sampling, lr=1e-2):
    def __init__(self, net: callable, sampled_points: jnp.array, lr=1e-2):
        self.net: Callable[[jnp.array], jnp.array] = net  # TODO: check if this is the right way to do it or use jnp.ndarray
        #self.n_sampling: int = n_sampling
        #self.lr: float = lr
        #self.sampler = DB_sampler(n_points=n_sampling)  # TODO: get rid of this because of cohesion
        #self.pg = PersistentGradient()  # TODO: get rid of this because of cohesion
        self.sampled_points = sampled_points
        #self.sampled_points = self.sampler.sample(0., net)   #TODO: should use the actual parameters of the net, not 0 !!!
                                                # This might be done by introducing accessor methods in FCNN and bumps.
        # TODO: Find out why this function takes a long time to run
    
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
        #print(self.sampled_points.shape, t.shape, self._normal_unit_vectors(theta).shape)
        #print("2", self.sampled_points + t * self._normal_unit_vectors(theta))
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
        #new_points = self._parametrization_normal_lines(t, theta)
        # res = self.sampler._loss(new_points, theta, self.net)
        #return res
        
        
        # Working solution for special case
        #print((self._parametrization_normal_lines(t, theta) ** 2).sum(axis=1).shape)
        #print(theta ** 2 * jnp.ones(self.sampled_points.shape[0]).shape)
        #print("3", (self._parametrization_normal_lines(t, theta) ** 2).sum(axis=1)\
        #    - theta ** 2 * jnp.ones(self.sampled_points.shape[0]))
        return (self._parametrization_normal_lines(t, theta) ** 2).sum(axis=1)\
            - theta ** 2 * jnp.ones(self.sampled_points.shape[0])  # should have shape (n_sampling, )
    
    #@implicit_diff.custom_root(_optimality_condition)
    def _inner_problem(self, t_init, theta): #, n_epochs=30, lr=1e-2):
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
        # new_points = self._degree_of_freedom(t, theta)
        # print("inner 2")
        # return self.sampler.sample(theta, self.net, new_points,
        #                            lr=lr, epochs=n_epochs, delete_outliers=False)   #TODO create a new method that only
        #                                                                             #optimizes t, instead of sample
        del t_init
        return jnp.zeros(self.sampled_points.shape[0])  # should have the shape (n_sampling, )
        #return jnp.zeros_like(self.sampled_points.shape[0])
        
    def t_star(self, theta):
        """ This function returns the optimal value of t, i.e., the value of t that
        minimizes the distance between the points and the decision boundary.
        """
        t_init = None
        #print("param normal lines : \n", self._parametrization_normal_lines(jnp.zeros(self.n_sampling), theta))
        return custom_root(self._optimality_condition)\
            (self._inner_problem)(t_init, theta)
    
    def update_sampled_points(self):
        pass
    
    ######################################################################
    
    # TODO: This should be outside the class
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
    
    # TODO: Put this in a separate optimization class
    def optimize(self, theta_init, n_epochs):
        """This is the main function of the class. It allows to optimize the
        topological loss of the decision boundary of net by modifying its parameter(s)
        theta. The gradient of the topological loss wrt the points is computed by the
        persistent gradient class, while the dependency of the points on the parameters
        of the network is computed by differentiating the output of the _inner_problem
        wrt theta.
        
        Note : a future improvement is to accept a loss function as an additional argument.
        The loss will be added to the topological loss in order to optimize not only the
        decision boundary, but also the accuracy of the network.

        Args:
            theta (jnp.array): the parameter we wish to optimize
            net (function): the function (neural network) parametrized by theta
        """
        
        theta = jnp.array(theta_init)
        optimizer = optax.adam(self.lr)
        params = {'theta': theta}
        opt_state = optimizer.init(params)
        #opt_state = optimizer.init(theta)
        loss = lambda x: self.toploss(x)     #in later versions this should include cross entropy
        
        for epoch in range(n_epochs):
            grads = grad(loss)(params['theta'])
            #grads = grad(loss)(theta)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            #theta = optax.apply_updates(theta, updates)
            
            # the points that were initially sampled by the sampler should be frequently updated.
            # every 5 epochs we set them so the value of new_points, that is, the intersections
            # of the normal lines with the DB given by the current value of theta.
            if epoch % 5 ==0:
                t = jnp.zeros_like(self.sampled_points[:, 0])
                self.sampled_points = self._inner_problem(t, theta=theta)

  