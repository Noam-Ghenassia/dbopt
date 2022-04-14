from jax import numpy as jnp
from jax import jit
from functools import partial
import numpy as np
import matplotlib.pyplot

class Bumps():
    """This class defines a function of 2 variables that is the sum of two gaussian kernels
    centered at (-1, 0) and (1, 0). The functions has a third parameter "theta" that is meant to be
    optimized in order to modify the homology of the graph's theta-level set.
    """
    
    @partial(jit, static_argnums=(0,))
    def level(self, x, theta=0):
        """The model function. It is the sum of two gaussian kernels centered at (0, -1) and (0, 1),
        to which we subtract (0.75 + theta). Theta can be viewed as a hyperparameter of the function,
        and changing its value modifies the homology of the 0-level set (by analogy to the weights of
        a neural network, that we can change to modify the homology of the decision boundary).

        Args:
            x (numpy.array): the points over which the function is evaluated.
            theta (int, optional): the hyperparameter that defines the 0-level set. Defaults to 0.

        Returns:
            np.array: the output of the function at inputs x.
        """
        a = jnp.exp(-1.5*((x[:, 0]-1)**2 + x[:, 1]**2))
        b = jnp.exp(-1.5*((x[:, 0]+1)**2 + x[:, 1]**2))
        return a+b-theta-.75
    
    def plot(self, figure, x_min=-2., x_max=2., y_min=-2., y_max=2.):
        """This function allows to plot the bumps function on a given figure.

        Args:
            figure (matplotlib.axes.Axes): the figure over which the function is plotted.
            x_min (float, optional): lower bound of the x axis of the plot. Defaults to -2..
            x_max (float, optional): high bound of the x axis of the plot. Defaults to 2..
            y_min (float, optional): lower bound of the y axis of the plot. Defaults to -2..
            y_max (float, optional): high bound of the y axis of the plot. Defaults to 2..
        """
        x = jnp.linspace(x_min, x_max, 200)
        y = jnp.linspace(y_min, y_max, 200)
        grid_x = jnp.meshgrid(x, y)[0].reshape(-1, 1)
        grid_y = jnp.meshgrid(x, y)[1].reshape(-1, 1)
        grid = jnp.concatenate([grid_x, grid_y], axis=1)
        
        out = self.level(grid).reshape(len(x), -1)
        figure.contourf(jnp.meshgrid(x, y)[0], jnp.meshgrid(x, y)[1], out)