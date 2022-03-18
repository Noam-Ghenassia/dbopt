from jax import numpy as jnp
from jax import random
import math
import matplotlib.pyplot as plt

class Spiral():
    
    def __init__(self, n_points):
        self.n_points = n_points
        self.data, self.labels = self.create_dataset()
    
    def _create_dataset(self, plot=False):
        
        key = random.PRNGKey(0)
        key1, key2 = random.split(key)
        
        theta1 = random.uniform(key=key1, shape=(self.n_points,), minval=0, maxval=1.5)
        noise1 = random.normal(key=key1, shape=theta1.shape)
        r1 = theta1+noise1
        x1 = r1*jnp.cos(theta1)
        y1 = r1*jnp.sin(theta1)
        C1 = jnp.concatenate((x1, y1), axis=1)
        C1 = jnp.concatenate([jnp.zeros((x1.shape[0], 1)), C1], axis=1)
        
        
        theta2 = random.uniform(key=key2, shape=(self.n_points,), minval=0, maxval=1.5)
        noise2 = random.normal(key=key2, shape=theta2.shape)
        r2 = theta2+noise2
        x2 = r2*jnp.cos(theta2 + math.pi)
        y2 = r2*jnp.sin(theta2 + math.pi)
        C2 = jnp.concatenate((x2, y2), axis=1)
        C2 = jnp.concatenate([jnp.ones((x2.shape[0], 1)), C2], 1)
        
        dataset = jnp.concatenate([C1, C2], axis=0)
        dataset = random.permutation(random.PRNGKey(0), dataset)
        data = dataset[:, 1:]
        labels = dataset[:, :1]
        labels = labels.reshape(-1)

        return data, labels
        
        
    def plot(self, figure):
        
        ind_0 = jnp.where(self.labels==0)
        ind_1 = jnp.where(self.labels==1)
        figure.plot(self.data[ind_0, 0], self.data[ind_0, 1], 'bo', self.data[ind_1, 0], self.data[ind_1, 1], 'ro')