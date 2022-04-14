import numpy as np
from jax import numpy as jnp
from jax import grad
from jax import jacfwd
#from jax.experimental.optimizers import adam
import optax
from jaxopt import implicit_diff


from dbopt import persistent_gradient as pg
from dbopt.DB_sampler import DB_sampler


class DB_Top_opt():
    """This class allows to modify the parameters of a function (typically, the
    weights of a neural network) in order to optimize the homology of the
    simplicial complex constructed on a set of points sampled from the 0-level
    set of that function (or the decision boundary of the neural network). The
    optimization minimizes a user provided function of the persistence diagram of
    the filtration of the point cloud.
    
    Note : a future improvement would be to add the ability for the user to only
    provide a set of Betti numbers instead of the actual function to optimize.
    """
    
    def __init__(self, net, lr=1e-2, num_epochs=150):
        self.net = net
        self.lr = lr
        self.num_epochs = num_epochs
        self.sampler = DB_sampler()
        self.pg = pg.PersistentGradient()
        self.x = self.sampler.sample(0., net)   # should use the actual parameters of the net, not 0 !!!
    
    def get_points(self):
        return self.x
        
    def _normal_unit_vectors(self, net, theta):
        """This function returns a set of vectors that are normal to the decision boundary
        at the points that were sampled from it by the sampler.

        Args:
            net (function): the network that is being optimized.

        Returns:
            jnp.array: an n*d matrix with rows the normal vectors of the decision boundary
            evaluated at the points sampled by the sampler.
        """
        normal_vectors = jacfwd(lambda x : net(x, theta))(self.x, theta)
        norms = jnp.linalg.norm(normal_vectors, axis=1)
        return jnp.divide(normal_vectors, norms[:, None])
        
    def _degree_of_freedom(self, t:jnp.array, net, theta):
        """This function gives a new set of points that depend on the points initially sampled
        by the sampler, but that only depend on a single real coordinate each. These new
        points are bound to move along a line that is normal to the decision boundary
        and passes through one of the original points (each).

        Args:
            net (function): the network that is being optimized
            t (jnp.array): the parameters that define the position of the new points.

        Returns:
            jnp.array: the new points.
        """
        return self.x + jnp.multiply(t, self._normal_unit_vectors(net, theta))

    def _optimality_condition(self, t, theta, net):
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
        new_points = self._degree_of_freedom(t, net, theta)
        return self.sampler._loss(new_points, theta, net)
    
    @implicit_diff.custom_root(_optimality_condition)
    def _inner_problem(self, t, theta, net, n_epochs=30, lr=1e-2):
        """This function is the inner optimization problem. It simply samples the
        decision boundary of the network, but with the custom root decorator it
        is possible to get the jacobian of the optimal points wrt the parameters
        of net (theta), which will be necessary in the chain rule that allows to differentiate
        the topological loss wrt theta.

        Args:
            t (jnp.array): the parameters that define the position of the new points
            theta (jnp.array): the points that sample the decision boundary
            net (function): the function parametrized by theta
        """
        new_points = self._degree_of_freedom(t, net, theta)
        return self.sampler.sample(net, new_points, lr=lr, epochs=n_epochs, delete_outliers=False)
        
    
    def toploss(self, theta, net):
        """This the topological loss that is optimized by the class. It depends on the
        value of theta.

        Args:
            theta (jnp.array): the parameter(s) of the funcrion of which we optimize the
            decision boundary's homology.

        Returns:
            jnp.array: the value of the topological loss.
        """
        t_init = jnp.zeros_like(self.x[:, 0])
        new_points = self._inner_problem(t_init, theta, net)
        return self.pg.single_cycle(new_points)
    
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
        loss = lambda params: self.toploss(params, self.net)     #in later versions this should include cross entropy
        
        for epoch in range(n_epochs):
            grads = grad(loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # the points that were initially sampled by the sampler should be frequently updated.
            # every 5 epochs we set them so the value of new_points, that is, the intersections
            # of the normal lines with the DB given by the current value of theta.
            if epoch % 5 ==0:
                self.x = params['theta']

        
       
       
