# Generic imports
import numpy as np

# A list of continuous environments and their action_space size
# - Pendulum-v0              (1)
# - MountainCarContinuous-v0 (1)
# - LunarLanderContinuous-v2 (2)
# - BipedalWalker-v2         (4)
# - BipedalWalkerHardcore-v2 (4)

# MountainCar-v0
# CartPole-v1

env_name     = 'CartPole-v1'

alg_type     = 'discrete' # 'discrete' or 'continuous'

n_episodes   = 300          # Max nb of episodes
n_steps      = 5000         # Max nb of steps per episode
n_avg        = 1            # Nb of runs for averaged results
render_every = 300          # Rendering frequency (in episodes)

actor_lr     = 1.0e-2       # Actor  learning rate
critic_lr    = 1.0e-3       # Critic learning rate
batch_size   = 64           # Batch size
n_epochs     = 1            # Nb of epochs for training
buff_size    = 4*batch_size # Size of buffer for training

clip         = 0.1          # Clipping parameter
entropy      = 0.01         # Entropy coefficient
gamma        = 0.99         # Discount coefficient
gae_lambda   = 0.99         # GAE coefficient
alpha        = 1.0          # Smooth update parameter
