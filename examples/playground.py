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


# %%
