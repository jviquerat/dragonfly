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

env_name     = "Pendulum-v0"

n_episodes   = 100
n_steps      = 1000
n_avg        = 1
render_every = 100

learn_rate   = 5.0e-3
batch_size   = 256
buff_size    = 4*batch_size
n_epochs     = 64

# PPO specific
clip         = 0.3
entropy      = 0.001
gamma        = 0.99
gae_lambda   = 0.95
update_alpha = 0.90
