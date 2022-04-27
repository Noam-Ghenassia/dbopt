# %%
import pwd
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


#%%
import numpy as np
import jax.numpy as jnp
import jax.random as random

from dbopt.DB_sampler import DB_sampler
from dbopt.Bumps import Bumps
from dbopt.Datasets import Spiral
from dbopt.FCNN import FCNN
from dbopt.DB_Top_opt import DB_Top_opt

#%%
bumps = Bumps()
opt = DB_Top_opt(bumps.level, n_sampling=20)
print(opt.toploss(theta=0))   #same error



# %%
from dbopt.ImpliciteDifferentiationSample import ImpliciteDifferentationSample as d1
#from dbopt.ImpliciteDifferentiationSample import DifferentiationSample as d2

from jax import grad
import jax.numpy as jnp
#c2=d2()
#print(grad(c2.f)(2.))
#print(jnp.exp(2.))

c1=d1()
a = jnp.array([2.0])
c1.implicite_differentiation(a)
# %%
import jax.numpy as jnp

from jax import grad
from dbopt.ImpliciteDifferentiationSample import ImpliciteDifferentationSample

# %%
d1 = ImpliciteDifferentationSample()

grad(d1.implicite_differentiation)(jnp.array([1.0]))
# %%

# %%