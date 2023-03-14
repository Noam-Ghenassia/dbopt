from jax import random
from jax import numpy as jnp
from jax import value_and_grad
#from jax.experimental.optimizers import adam
from optax import adam, apply_updates
import numpy as np

class DecisionBoundarySampler():
    """This class allows to sample the descision boundary of a neural network. The number of points
    randomly sampled from the latent space should be inputed at class istanciation, while the
    network whose decision boundary is to be sampled is passed as an argument to the sample function.
    """

    def __init__(self, n_points=1000,
                 input_dim=2, min_x=-2., max_x=2., min_y=-2., max_y=2.):
        
        self.key = random.PRNGKey(0)
        self.n_points = n_points
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.input_dim = input_dim
        self.points = self._random_sampling()
    
    def _random_sampling(self):
        """Creates a random sampling of the latent in the rectangle spanned by the min and max vectors.

        Returns:
            jax.numpy.array: the points sampled from the latent space
        """
        
        # shape = (self.n_points, self.input_dim)
        # return random.uniform(self.key, shape=shape, minval=self.min_x, maxval=self.max_x)
        self.key, x_key = random.split(self.key)
        x = jnp.expand_dims(random.uniform(x_key, minval=self.min_x, maxval=self.max_x, shape=(self.n_points,)), axis=1)
        y = jnp.expand_dims(random.uniform(self.key, minval=self.min_y, maxval=self.max_y, shape=(self.n_points,)), axis=1)
        return jnp.concatenate((x, y), axis=1)

    def _loss(self, points, theta, net,  squaredDiff=True, return_losses=False):
        """A loss function that penalises points that lie far from the DB.

        Args:
            squaredDiff (bool, optional): Squared (True) or absolute (False) difference of the logits. Defaults to False.
            return_losses (bool, optional): Set to True to get the individual losses. Defaults to False.

        Returns:
            jax.numpy.ndarray: the loss value
        """

        #preds = net(points, theta)     #TODO: add an apply method to bumpy so we can use it the same way
        preds = net.apply(theta, points)
        
        if preds.ndim == 2:
            if squaredDiff:
                losses = (preds[:, 0]-preds[:, 1])**2
            else :
                losses = jnp.abs(preds[:, 0]-preds[:, 1])
        else :
            # We add this for the case where net isn't a neural network. In this case, the loss is
            # minimized wherever net outputs 0, i.e., at the 0-level set of net.
            if squaredDiff:
                losses = preds**2
            else :
                losses = jnp.abs(preds)
        
        if return_losses :
            return jnp.mean(losses), losses
        return jnp.mean(losses)

    #def _step(self, net, theta, epoch, opt_state, opt_update, get_points):
    def _step(self, net, theta, opt, opt_state, return_points=False):
        """Performs an optimization step.
        """

        # value, grads = value_and_grad(lambda x: self._loss(x, theta, net))(get_points(opt_state))
        # opt_state = opt_update(epoch, grads, opt_state)
        # return value, opt_state

        loss, grads = value_and_grad(lambda x: self._loss(x, theta, net))(self.points)
        updates, opt_state = opt.update(grads, opt_state)
        self.points = apply_updates(self.points, updates)
        return loss
    
    def get_points(self):
        return self.points
    
    def sample(self, theta, net, points=None, lr=1e-2, epochs=1000, threshold=.2, delete_outliers=True):
        """The main function of the class. It samples the decision boundary of the
        network passed as argument to the DB_sampler.

        Args:
            net (function): the network of which the decision boundary is to be sampled.
            points(jnp.array): if None, then the points pushed to the DB are the class attribute 'points'. Else, the
            provided points are pushed to the DB.
            lr (float, optional): the learning rate applied by the optimizer. Defaults to 1e-2.
            epochs (int, optional): the number of optimization epochs. Defaults to 1000.
            threshold (float, optional): the loss value above which the points are deleted after training. Defaults to .2.
            delete_outliers (bool, optional): set to True to delete points with a loss value above
            the indicated threshold. Defaults to True.

        Returns:
            jax.numpy.ndarray: a sampling of the decision boundary obtained by optimizing a loss that compares the logits.
        """

        if points is None:
            # opt_init, opt_update, get_points = adam(lr)
            # opt_state = opt_init(self.points)
            opt = adam(lr)
            opt_state = opt.init(self.points)
            
            losses = np.empty((epochs,))

            for epoch in range(epochs):
                # value, opt_state = self._step(net, theta, epoch, opt_state, opt_update, get_points)
                # self.points = get_points(opt_state)
                # losses[epoch] = value
                value = self._step(net, theta, opt, opt_state)
                losses[epoch] = value
            
            #DELETE POINTS WITH HIGH LOSS
            if delete_outliers:
                _, losses = self._loss(self.points, theta, net, return_losses=True)
                indices = np.argwhere(losses > threshold)
                self.points = np.delete(self.points, indices, axis=0)
            
            return self.points
        
        else :
            # opt_init, opt_update, get_points = adam(lr)
            # opt_state = opt_init(points)
            self.points=points
            opt = adam(lr)
            opt_state = opt.init(self.points)

            losses = np.empty((epochs,))

            for epoch in range(epochs):
                # value, opt_state = self._step(net, theta, epoch, opt_state, opt_update, get_points)
                # points = get_points(opt_state)
                # losses[epoch] = value
                value = self._step(net, theta, opt, opt_state)
                losses[epoch] = value
            
            #DELETE POINTS WITH HIGH LOSS
            if delete_outliers:
                _, losses = self._loss(points, theta, net, return_losses=True)
                indices = np.argwhere(losses > threshold)
                points = np.delete(points, indices, axis=0)
            
            return self.points