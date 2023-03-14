import functools
from typing import Any, Callable, Sequence, Optional

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state
import jax
from jax import lax, random, numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax



class FCNN(nn.Module):
    """A fully connected neural network. When instanciated, it
    should be initialized with a dummy input with the desired
    shape, e.g., params = network.init(init_key, x_init).
    """
    num_neurons_per_layer: Sequence[int]
    #layers: Sequence[Callable]

    def setup(self):
        """This function generates the structure of the network.
        It is automatically called when an FCNN object is instaciated.
        """
        self.layers = [nn.Dense(n) for n in self.num_neurons_per_layer]

    def __call__(self, x):
        """This function gives the output of the network for input x.

        Args:
            x (jnp.array): The input.

        Returns:
            jnp.array: The output.
        """
        activation = x
        for i, layer in enumerate(self.layers):
            activation = layer(activation)
            if i != len(self.layers) - 1:
                activation = nn.elu(activation)
        # return jnp.exp(nn.log_softmax(activation))
        return nn.softmax(activation)
    
    def make_loss_fn(self, data, labels):
        """This function allows to create a categorical cross-entropy
        loss function that is evaluated over a given set of data points
        and corresponding labels.

        Args:
            data (jnp.array): The datapoints.
            labels (jnp.array): The corresponding labels.
            
        Returns:
            Callable: the cross-entropy loss function.
        """
        
        def loss_fn(params):
            preds = self.apply(params, data)
            one_hot_gt_labels = jax.nn.one_hot(labels, num_classes=2)
            loss = -jnp.mean(jnp.sum(one_hot_gt_labels * jnp.log(preds), axis=-1))
            return loss
        
        return loss_fn
    
    def make_batches(self, key, dataset, batch_size=64):
        """This function allows to partition the dataset into batches with
        specified size. 

        Args:
            dataset (jnp.array): The dataset that is partitionned.
            batch_size (int, optional): The size of the returned batches. Defaults to 64.

        Returns:
            tuple: The batches.
        """
        n_points = dataset.shape[0]
        remainder = n_points % batch_size
        num_full_batches = (n_points - remainder)/batch_size
        permuted_dataset = random.permutation(key, dataset)
        batches_list = []
        for batch in range(int(num_full_batches)):
            batches_list.append(permuted_dataset[batch:batch+batch_size, :])
        batches_list.append(permuted_dataset[-remainder-1:-1, :])
        return batches_list
    
    def _make_optimizer(self, params, lr):
        opt = optax.adam(learning_rate=lr)
        opt_state = opt.init(params)
        return opt, opt_state
    
    def train(self, key, params,  dataset, epochs, lr=0.01, logs_frequency=0, test_set=None):
        """This function allows to rain the neural network. The parameters
        of the network should be passed as argument after initialization.

        Args:
            params (jax.FrozenDict): The parameters of the network.
            dataset (jnp.array): The training set.
            epochs (int): The number of training epochs.
            logs_frequency (int): The frequency (in epochs) at which logs are printed
            test_set (jnp.array): The data on which the test loss printed in the logs is computed.

        Returns:
            jax.FrozenDict: The network's parameters after training.
        """
        optimizer, opt_state = self._make_optimizer(params, lr)
        for epoch in range(epochs):
            key, batch_key = random.split(key)
            batches = self.make_batches(batch_key, dataset)
            epoch_loss = jnp.empty((len(batches),))
            for batch_nb, batch in enumerate(batches):
                batch_data = batch[:, 1:]
                batch_labels = jnp.squeeze(batch[:, :1], axis=1)
                loss_fn = self.make_loss_fn(batch_data, batch_labels)
                loss_fn, grads = jax.value_and_grad(loss_fn)(params)
                epoch_loss = epoch_loss.at[batch_nb].set(loss_fn)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
            
            if (logs_frequency != 0) and (epoch % logs_frequency == 0):
                if test_set is None:
                    print(f'epoch {epoch}, loss = {loss_fn}, training accuracy = {self.accuracy(params, dataset)}')
                else:
                    print(f'epoch {epoch}, loss = {loss_fn}, training accuracy = {self.accuracy(params, dataset)}, test accuracy = {self.accuracy(params, test_set)}')
        
        return params
    
    def accuracy(self, params, dataset):
        data = dataset[:, 1:]
        labels = dataset[:, :1]
        preds = jnp.round(self.apply(params, data))[:, 1]
        return (jnp.dot(preds, labels) + jnp.dot(1-preds, 1-labels))/labels.shape[0]

    
    def plot_decision_boundary(self, params, ax, x_min=-10., x_max=10., y_min=-10., y_max=10.):

        x = jnp.linspace(x_min, x_max, 220)
        y = jnp.linspace(y_min, y_max, 220)
        grid_x = jnp.meshgrid(x, y)[0].reshape(-1, 1)
        grid_y = jnp.meshgrid(x, y)[1].reshape(-1, 1)
        grid = jnp.concatenate([grid_x, grid_y], axis=1)

        grid = jnp.expand_dims(grid, axis=1)
        out = self.apply(params, grid)
        out = np.squeeze(out, axis=1)[: , 1].reshape(len(x), -1)

        ax.contourf(jnp.meshgrid(x, y)[0], jnp.meshgrid(x, y)[1], out)

