import time
import numpy as np
import numpy.random as npr
import itertools
import jax.numpy as jnp
from jax import random
from jax.experimental import stax
from jax.experimental import optimizers
from jax import jit, grad
from jax.nn import softmax
#from jax.nn import Elu, log_softmax
#from flax.linen import Dense
from jax.experimental.stax import Dense, Relu, LogSoftmax, Tanh, Sigmoid, Elu


class FCNN():
    """A fully connected binary classifier with 5 hidden layers and Elu activation functions.
    """
    
    def __init__(self, step_size = 0.01, num_epochs = 500, batch_size = 4,
                 momentum_mass = 0.9):

        self.init_random_params, self.predict = stax.serial(
            Dense(10), Elu,
            Dense(10), Elu,
            Dense(10), Elu,
            Dense(10), Elu,
            Dense(10), Elu,
            Dense(2), LogSoftmax)  #log_softmax
        self.params = []
        self.rng = random.PRNGKey(0)
        self.step_size = step_size,
        self.num_epochs = num_epochs,
        self.batch_size = batch_size,
        self.momentum_mass = momentum_mass
    
    
    def _loss(self, params, batch):
        """The loss function that is optimized during training.

        Args:
            params : the parameters of the network
            batch : the batch over which the loss is to be computed

        Returns:
            float: the loss value
        """
        inputs, targets = batch
        preds = self.predict(params, inputs)
        return -jnp.mean(preds * targets)

    def _accuracy(self, params, batch):
        """This function computes the accuracy of the network over a given dataset.

        Args:
            params : the parameters of the network
            batch (_type_): the data over which the accuracy is computed

        Returns:
            float: the accuracy of the network
        """
        inputs, targets = batch
        #targets = targets[0, :]
        target_class = np.argmax(targets)
        predicted_class = np.argmax(self.predict(params, inputs), axis=1)
        return jnp.mean(predicted_class == target_class)

    def _one_hot(self, labels):
        """This function allows to encode the labels in one-hot fashion.

        Args:
            labels (jax.numpy.array): the labels

        Returns:
            jax.numpy.array: the one-hot encoded labels
        """
        C2 = (jnp.ones_like(labels)-labels).reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        return jnp.concatenate([labels, C2], axis=1)
    
    def _data_stream(self, num_train, num_batches, data, labels, rng):
        """creates batches with desired batch size by randomly permuting the dataset.

        Args:
            num_train (int): the number of training example
            num_batches (int): the number of batches that should be created
            data (jax.numpy.array): the samples
            labels (jax.numpy.array): the labels
            rng (jaxlib.xla_extension.DeviceArray): a random number generator

        Yields:
            jax.numpy.array: a batch
        """
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * self.batch_size[0]:(i + 1) * self.batch_size[0]]
                yield data[batch_idx], self._one_hot(labels[batch_idx])
    
    def _update(self, i, opt_state, batch, opt_update, get_params):
        """This function updates the parameters of the network.

        Args:
            i (int): the current training epoch
            opt_state (jax.example_libraries.optimizers.OptimizerState): the current state of the network
            batch (jax.numpy.array): the batch used for the optimization step
            opt_update (function): _description_
            get_params (function): accessor function for the network's parameters

        Returns:
            _type_: _description_
        """
        self.params = get_params(opt_state)
        return opt_update(i, grad(lambda x : self._loss(x, batch))(self.params), opt_state)
    
    def plot_decision_boundary(self, ax, x_min=-10., x_max=10., y_min=-10., y_max=10.):
        """This function allows to plot the network's decision boundary on a given figure.

        Args:
            ax (matplotlib.axes.Axes): the pyplot figure on which the dataset is plotted.
            x_min (float, optional): lower bound of the x axis of the plot. Defaults to -10..
            x_max (float, optional): high bound of the x axis of the plot. Defaults to 10..
            y_min (float, optional): lower bound of the y axis of the plot. Defaults to -10..
            y_max (float, optional): high bound of the y axis of the plot. Defaults to 10..
        """
        x = jnp.linspace(x_min, x_max, 220)
        y = jnp.linspace(y_min, y_max, 220)
        grid_x = jnp.meshgrid(x, y)[0].reshape(-1, 1)
        grid_y = jnp.meshgrid(x, y)[1].reshape(-1, 1)
        grid = jnp.concatenate([grid_x, grid_y], axis=1)

        grid = jnp.expand_dims(grid, axis=1)
        out = self.predict(self.params, grid)
        out = np.squeeze(out, axis=1)
        out = softmax(out, axis=1)[: , 1].reshape(len(x), -1)
        
        ax.contourf(jnp.meshgrid(x, y)[0], jnp.meshgrid(x, y)[1], out)

    def train(self, data, labels):
        """The main function of the class. It trains the neural network on the dataset
        consisting of data and labels.

        Args:
            data (jax.numpy.array): the samples
            labels (jax.numpy.array): the labels
        """
        rng = npr.RandomState(0)
        key = random.PRNGKey(2)
        num_train = data.shape[0]
        num_complete_batches, leftover = divmod(num_train, self.batch_size[0])
        num_batches = num_complete_batches + bool(leftover)
        batches = self._data_stream(num_train, num_batches, data, labels, rng)
        
        input_shape = (-1, 2)
        opt_init, opt_update, get_params = optimizers.adam(self.step_size[0])
        _, init_params = self.init_random_params(key, input_shape)
        opt_state = opt_init(init_params)
        itercount = itertools.count()
        
        print("\nStarting training...")
        for epoch in range(self.num_epochs[0]):
            start_time = time.time()
            for _ in range(num_batches):
                opt_state = self._update(next(itercount), opt_state, next(batches),
                                         opt_update, get_params)
                self.params = get_params(opt_state)
            epoch_time = time.time() - start_time

            if epoch%50 == 0 :
                print('Epoch : ', epoch)
                train_acc = self._accuracy(self.params, (data, self._one_hot(labels)))
                #test_acc = accuracy(params, (test_data, one_hot(test_labels)))
                print('Epoch : ', epoch, ', Train accuracy : ', train_acc)
                
