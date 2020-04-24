# Generic imports
import numpy as np

# Choose actor
actor        = 'ppo-cma' # 'ppo' or 'ppo-cma'

# Generic parameters
n_params     = 2
init_obs     = np.zeros(n_params)
x_min        =-5.0
x_max        = 5.0
y_min        =-5.0
y_max        = 5.0

#x_min        =-2.0
#x_max        = 2.0
#y_min        =-1.0
#y_max        = 3.0

n_gen        = 30
n_ind        = 6
n_avg        = 10

learn_rate   = 5.0e-3
actor_epochs = 64

# PPO specific
clip         = 0.5
entropy      = 0.005

# PPO-CMA specific
n_batch      = 5
clip_adv     = True
mirror_adv   = True
