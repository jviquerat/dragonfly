# Generic imports
import numpy as np

# A list of continuous environments and their action_space size
# - Pendulum-v0              (1)
# - MountainCarContinuous-v0 (1)
# - LunarLanderContinuous-v2 (2)
# - BipedalWalker-v2         (4)
# - BipedalWalkerHardcore-v2 (4)

env_name     = "MountainCarContinuous-v0"

n_episodes   = 1000 # Max nb of episodes
n_steps      = 5000 # Max nb of steps per episode
n_avg        = 1    # Nb of runs for averaged results
render_every = 30   # Rendering frequency (in episodes)

learn_rate   = 1.0e-3       # Learning rate
batch_size   = 512          # Batch size
n_epochs     = 10           # Nb of epochs for training
buff_size    = 1*batch_size # Size of buffer for training

clip         = 0.1
entropy      = 0.001
gamma        = 0.99
gae_lambda   = 0.95
update_alpha = 0.90
