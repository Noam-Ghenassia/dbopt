from abc import ABC, abstractmethod
from jax import numpy as jnp
from jax import random
import math
import matplotlib.pyplot as plt

class Dataset_2D(ABC):
    """Abstract class for 2-classes datasets in 2 dimensions.

    n_points (int) : the number of samples in each class.
    """
    def __init__(self, n_points, key):
        self.n_points = n_points
        self.key = key
        #self.data, self.labels = self._create_dataset()
        self.dataset = self._create_dataset()
        super().__init__()
    
    @abstractmethod
    def _create_dataset(self):
        pass

    def plot(self, figure):
        """This methods plots the dataset on the figure passed as argument.

        Args:
            figure (matplotlib.axes.Axes): the pyplot figure on which the dataset is plotted.
        """
        data = self.dataset[:, 1:]
        labels = jnp.squeeze(self.dataset[:, :1], axis=1)
        ind_0 = jnp.where(labels==0)
        ind_1 = jnp.where(labels==1)
        figure.plot(data[ind_0, 0], data[ind_0, 1], 'bo',
                    data[ind_1, 0], data[ind_1, 1], 'ro')

    def get_dataset(self):
        """Accessor method.

        Returns:
            (jnp.array, jnp.array): the data and labels of the dataset.
        """
        #return self.data, self.labels
        return self.dataset

# def split_dataset(dataset, test_size: float=0.2) -> Tuple[dataset, datset]:
    
#     return train_ds, test_ds

class Spiral(Dataset_2D):

    def __init__(self, n_points, key):
        super().__init__(n_points, key)
    
    def _create_dataset(self):
        
        #key = random.PRNGKey(0)
        key1, key2 = random.split(self.key)
        
        key11, key12 = random.split(key1)
        theta1 = 2*math.pi*random.uniform(key=key11, shape=(self.n_points,), minval=0, maxval=1.5)
        noise1 = 0.2*random.normal(key=key12, shape=theta1.shape)
        r1 = theta1+noise1
        x1 = jnp.expand_dims(r1*jnp.cos(theta1), axis=1)
        y1 = jnp.expand_dims(r1*jnp.sin(theta1), axis=1)
        C1 = jnp.concatenate((x1, y1), axis=1)
        C1 = jnp.concatenate([jnp.zeros((x1.shape[0], 1)), C1], axis=1)
        
        key21, key22 = random.split(key2)
        theta2 = 2*math.pi*random.uniform(key=key21, shape=(self.n_points,), minval=0, maxval=1.5)
        noise2 = 0.2*random.normal(key=key22, shape=theta2.shape)
        r2 = theta2+noise2
        x2 = jnp.expand_dims(r2*jnp.cos(theta2 + math.pi), axis=1)
        y2 = jnp.expand_dims(r2*jnp.sin(theta2 + math.pi), axis=1)
        C2 = jnp.concatenate((x2, y2), axis=1)
        C2 = jnp.concatenate([jnp.ones((x2.shape[0], 1)), C2], 1)
        
        dataset = jnp.concatenate([C1, C2], axis=0)
        dataset = random.permutation(random.PRNGKey(0), dataset)
        #data = dataset[:, 1:]
        #labels = dataset[:, :1]
        #labels = labels.reshape(-1)

        #return data, labels
        return dataset
