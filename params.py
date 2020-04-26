# Generic imports
import numpy as np

# A list of continuous environments and their action_space size
# - Pendulum-v0              (1)
# - MountainCarContinuous-v0 (1)
# - LunarLanderContinuous-v2 (2)
# - BipedalWalker-v2         (4)
# - BipedalWalkerHardcore-v2 (4)

env_name     = "LunarLanderContinuous-v2"

n_episodes   = 1000
n_steps      = 100
n_avg        = 1
render_every = 10

learn_rate   = 1.0e-4
batch_size   = 1024
n_epochs     = 4
buff_size    = 1*batch_size

clip         = 0.1
entropy      = 0.001
gamma        = 0.999
gae_lambda   = 0.95
update_alpha = 0.90
