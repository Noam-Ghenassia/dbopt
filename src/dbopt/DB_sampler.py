from jax import random
from jax import numpy as jnp
from jax import value_and_grad
from jax.experimental.optimizers import adam
import numpy as np

class DB_sampler():
    """This class allows to sample the descision boundary of a neural network. The number of points
    randomly sampled from the latent space should be inputed at class istanciation, while the
    network whose decision boundary is to be sampled is passed as an argument to the sample function.
    """

    def __init__(self, n_points=1000, epochs=1000,
                 input_dim=2, min=-2., max=2.):
        
        self.key = random.PRNGKey(0)
        self.n_points = n_points
        self.min_x = min
        self.max_x = max
        self.input_dim = input_dim
        self.points = self._random_sampling()
    
    def _random_sampling(self):
        """Creates a random sampling of the latent in the rectangle spanned by the min and max vectors.

        Returns:
            jax.numpy.array: the points sampled from the latent space
        """
        
        shape = (self.n_points, self.input_dim)
        return random.uniform(self.key, shape=shape, minval=self.min_x, maxval=self.max_x)

    def _loss(self, points, net, squaredDiff=False, return_losses=False):
        """A loss function that penalises points that lie far from the DB.

        Args:
            squaredDiff (bool, optional): Squared (True) or absolute (False) difference of the logits. Defaults to False.
            return_losses (bool, optional): Set to True to get the individual losses. Defaults to False.

        Returns:
            jax.numpy.ndarray: the loss value
        """
        logits = net(points)
        
        if logits.ndim == 2:
            if squaredDiff:
                losses = (logits[:, 0]-logits[:, 1])**2
            else :
                losses = jnp.abs(logits[:, 0]-logits[:, 1])
        else :
            # We add this for the case where net isn't a neural network. In this case, the loss is
            # minimized wherever net outputs 0, i.e., at the 0-level set of net.
            if squaredDiff:
                losses = logits**2
            else :
                losses = jnp.abs(logits)
        
        if return_losses :
            return jnp.mean(losses), losses
        return jnp.mean(losses)

    def _step(self, net, epoch, opt_state, opt_update, get_points):
        """Performs an optimization step.
        """
        
        value, grads = value_and_grad(lambda x: self._loss(x, net))(get_points(opt_state))
        opt_state = opt_update(epoch, grads, opt_state)
        return value, opt_state
    
    def get_points(self):
        return self.points
    
    def sample(self, net, lr=1e-2, epochs=1000, threshold=.2, delete_outliers=True):
        """The main function of the class. It samples the decision boundary of the
        network passed as argument to the DB_sampler.

        Args:
            net (function): the network of which the decision boundary is to be sampled.
            lr (float, optional): the learning rate applied by the optimizer. Defaults to 1e-2.
            epochs (int, optional): the number of optimization epochs. Defaults to 1000.
            threshold (float, optional): the loss value above which the points are deleted after training. Defaults to .2.
            delete_outliers (bool, optional): set to True to delete points with a loss value above
            the indicated threshold. Defaults to True.

        Returns:
            jax.numpy.ndarray: a sampling of the decision boundary obtained by optimizing a loss that compares the logits.
        """
        
        opt_init, opt_update, get_points = adam(lr)
        opt_state = opt_init(self.points)
        
        losses = np.empty((epochs,))

        for epoch in range(epochs):
            value, opt_state = self._step(net, epoch, opt_state, opt_update, get_points)
            self.points = get_points(opt_state)
            losses[epoch] = value
        
        #DELETE POINTS WITH BIG LOSS
        if delete_outliers:
            _, losses = self._loss(self.points, net, return_losses=True)
            indices = np.argwhere(losses > threshold)
            self.points = np.delete(self.points, indices, axis=0)
        
        return self.points