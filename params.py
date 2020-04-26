# Generic imports
import numpy as np

# Choose actor
actor        = 'ppo' # 'ppo' or 'ppo-cma'

# Generic parameters
#n_params     = 2
#init_obs     = np.zeros(n_params)
#x_min        =-5.0
#x_max        = 5.0
#y_min        =-5.0
#y_max        = 5.0

#x_min        =-2.0
#x_max        = 2.0
#y_min        =-1.0
#y_max        = 3.0

# A list of continuous environments and their action_space size
# - Pendulum-v0              (1)
# - MountainCarContinuous-v0 (1)
# - LunarLanderContinuous-v2 (2)
# - BipedalWalker-v2         (4)
# - BipedalWalkerHardcore-v2 (4)

env_name     = "MountainCarContinuous-v0"

n_episodes   = 1000
n_steps      = 100
n_avg        = 1
render_every = 10

learn_rate   = 1.0e-4
batch_size   = 32
n_epochs     = 8
buff_size    = n_epochs*batch_size

# PPO specific
clip         = 0.2
entropy      = 0.001
gamma        = 0.99
gae_lambda   = 0.98
update_alpha = 0.90
