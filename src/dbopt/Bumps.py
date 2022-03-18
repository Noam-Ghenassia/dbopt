from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot

class Bumps():
    
    def level(self, x, theta=0):
        a = jnp.exp(-1.5*((x[:, 0]-1)**2 + x[:, 1]**2))
        b = jnp.exp(-1.5*((x[:, 0]+1)**2 + x[:, 1]**2))
        return a+b-theta
    
    def plot(self, figure, x_min=-2., x_max=2., y_min=-2., y_max=2.):
        x = jnp.linspace(x_min, x_max, 200)
        y = jnp.linspace(y_min, y_max, 200)
        grid_x = jnp.meshgrid(x, y)[0].reshape(-1, 1)
        grid_y = jnp.meshgrid(x, y)[1].reshape(-1, 1)
        grid = jnp.concatenate([grid_x, grid_y], axis=1)
        
        out = self.level(grid).reshape(len(x), -1)
        figure.contourf(jnp.meshgrid(x, y)[0], jnp.meshgrid(x, y)[1], out)