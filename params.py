# Generic imports
import numpy as np

# A list of continuous environments and their action_space size
# - Pendulum-v0              (1)
# - MountainCarContinuous-v0 (1)
# - LunarLanderContinuous-v2 (2)
# - BipedalWalker-v2         (4)
# - BipedalWalkerHardcore-v2 (4)

# MountainCar-v0
# CartPole-v0
# Acrobot-v1
# LunarLander-v2

env_name     = 'CartPole-v0'
ep_end       = 200          # Nb of max episode steps
n_episodes   = 300          # Max nb of episodes
n_avg        = 5            # Nb of runs for averaged results
render_every = 10          # Rendering frequency (in episodes)

actor_lr     = 1.0e-2       # Actor  learning rate
critic_lr    = 1.0e-2       # Critic learning rate
batch_size   = 64           # Batch size
n_epochs     = 16           # Nb   of epochs  for training
buff_size    = 128          # Size of buffer  for training
n_buff       = 4            # Nb   of buffers for training

actor_arch   = [16,16]      # Actor  hidden layers
critic_arch  = [16,16]      # Critic hidden layers

pol_clip     = 0.1          # Policy   clip
grd_clip     = 0.5          # Gradient clip
entropy      = 0.01         # Entropy  coefficient
gamma        = 0.99         # Discount coefficient
gae_lambda   = 0.97         # GAE      coefficient
adv_clip     = True         # True for advantage clipping
bootstrap    = True         # True to bootstrap ending episodes
