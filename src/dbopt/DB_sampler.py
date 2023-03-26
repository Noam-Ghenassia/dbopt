from jax import random
from jax import numpy as jnp
from jax import value_and_grad
from optax import adam, apply_updates
import numpy as np

class DecisionBoundarySampler():
    """This class allows to sample the descision boundary of a neural network. The number of points
    randomly sampled from the latent space should be inputed at class istanciation, while the
    network whose decision boundary is to be sampled is passed as an argument to the sample function.
    """

    def __init__(self, n_points=1000,
                 input_dim=2, min=-1., max=1.):
        
        self.key = random.PRNGKey(0)
        self.n_points = n_points
        self.min = min
        self.max = max
        self.input_dim = input_dim
        self.points = self._random_sampling()
    
    def _random_sampling(self):
        """Creates a random sampling of the latent in the rectangle spanned by the min and max vectors.

        Returns:
            jax.numpy.array: the points sampled from the latent space.
        """
        
        self.key, sampling_key = random.split(self.key)
        return random.uniform(sampling_key, shape=(self.n_points, self.input_dim), minval=self.min, maxval=self.max)

    def _loss(self, points, theta, net, squaredDiff=True, return_losses=False, binary_classification=True):
        """A loss function that penalises points that lie far from the DB.

        Args:
            squaredDiff (bool, optional): Squared (True) or absolute (False) difference of the logits. Defaults to False.
            return_losses (bool, optional): Set to True to get the individual losses. Defaults to False.
            binary_classification (bool, optional): Set to False if the network outputs more than two values. Defaults to True.

        Returns:
            jax.numpy.ndarray: the loss value
        """

        preds = net.apply(theta, points)

        if binary_classification:
            if squaredDiff:
                losses = (preds[:, 0]-preds[:, 1])**2
            else:
                losses = jnp.abs(preds[:, 0]-preds[:, 1])
        else:
            raise NotImplementedError("The multiclass case is not implemented yet")
        
        if return_losses :
            return jnp.mean(losses), losses
        return jnp.mean(losses)

    def _step(self, net, theta, opt, opt_state, return_points=False):
        """Performs an optimization step.
        """
        
        loss, grads = value_and_grad(lambda x: self._loss(x, theta, net))(self.points)
        updates, opt_state = opt.update(grads, opt_state)
        self.points = apply_updates(self.points, updates)
        
        if return_points:
            return loss, self.points
        return loss
    
    def get_points(self):
        return self.points
    
    def sample(self, theta, net, points=None, lr=1e-2, epochs=1000, threshold=.2, delete_outliers=True):
        """The main function of the class. It samples the decision boundary of the
        network passed as argument to the DB_sampler.

        Args:
            net (function): the network of which the decision boundary is to be sampled.
            points(jnp.array): if None, then the points pushed to the DB are the class attribute 'points'. Else, the
                provided points are pushed to the DB and are ffected to the class attribute 'points.
            lr (float, optional): the learning rate applied by the optimizer. Defaults to 1e-2.
            epochs (int, optional): the number of optimization epochs. Defaults to 1000.
            threshold (float, optional): the loss value above which the points are deleted after training. Defaults to .2.
            delete_outliers (bool, optional): set to True to delete points with a loss value above
                the indicated threshold. Defaults to True.

        Returns:
            jax.numpy.ndarray: a sampling of the decision boundary obtained by optimizing a loss that compares the logits.
        """

        if points is None:
            opt = adam(lr)
            opt_state = opt.init(self.points)
            
            losses = np.empty((epochs,))

            for epoch in range(epochs):
                value = self._step(net, theta, opt, opt_state)
                losses[epoch] = value
            
            #DELETE POINTS WITH HIGH LOSS
            if delete_outliers:
                _, losses = self._loss(self.points, theta, net, return_losses=True)
                indices = np.argwhere(losses > threshold)
                self.points = np.delete(self.points, indices, axis=0)
            
            return self.points
        
        else :
            self.points=points
            opt = adam(lr)
            opt_state = opt.init(self.points)

            losses = np.empty((epochs,))

            for epoch in range(epochs):
                value = self._step(net, theta, opt, opt_state)
                losses[epoch] = value
            
            #DELETE POINTS WITH HIGH LOSS
            if delete_outliers:
                _, losses = self._loss(points, theta, net, squaredDiff=False, return_losses=True)
                indices = np.argwhere(losses > threshold)
                points = np.delete(points, indices, axis=0)
            
            return self.points