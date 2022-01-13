# Generic imports
import math
import random
import numpy as np

# Custom imports
from dragonfly.src.value.value import *
from dragonfly.src.decay.decay import *
from dragonfly.src.utils.error import *
from dragonfly.src.retrn.retrn import *

###############################################
### DQN agent
class dqn():
    def __init__(self, obs_dim, act_dim, n_cpu, pms):

        # Initialize from arguments
        self.name        = 'dqn'
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.n_cpu       = n_cpu
        self.gamma       = pms.gamma
        self.target_freq = pms.target_freq

        # Counter for target initialization
        self.tgt_update  = 0

        # Check n_cpu
        if (n_cpu != 1):
            error("dqn", "__init__",
                  "dqn agent does not support n_cpu > 1")

        # Initialize random limit
        self.eps = decay_factory.create(pms.exploration.type,
                                        pms = pms.exploration)

        # pol_dim is the true dimension of the action provided to the env
        # As dqn is only for discrete actions, it is set to 1
        self.pol_dim = 1

        # Build values
        if (pms.value.type != "q_value"):
            error("dqn", "__init__",
                  "Value type for dqn agent is not q_value")

        self.q_val = val_factory.create(pms.value.type,
                                        obs_dim = obs_dim,
                                        act_dim = act_dim,
                                        pms     = pms.value)
        self.q_tgt = val_factory.create(pms.value.type,
                                        obs_dim = obs_dim,
                                        act_dim = act_dim,
                                        pms     = pms.value)

    # Get actions
    def get_actions(self, obs):

        # "obs" possibly contains observations from multiple parallel
        # environments. We assume it does and unroll it in a loop
        act = np.zeros([self.n_cpu, self.pol_dim], dtype=int)

        p = random.uniform(0, 1)
        if (p < self.eps.get()):
            act = random.randrange(0, self.act_dim)
        else:
            obs = tf.cast([obs], tf.float32)
            val = self.q_val.get_values(obs)
            act = np.argmax(val)
            act = np.reshape(act, (-1))

        return act

    # Compute target
    def compute_target(self, obs, nxt, act, rwd, trm, bts):

        tgt = self.q_tgt.get_values(nxt)
        tgt = tf.reduce_max(tgt,axis=1)
        tgt = tf.reshape(tgt, [-1,1])
        tgt = rwd + trm*self.gamma*tgt

        return tgt

    # Update target
    def update_target(self):

        # Update target if necessary
        if (self.tgt_update == self.target_freq):
            self.q_tgt.set_weights(self.q_val.get_weights())
            self.tgt_update  = 0
        else:
            self.tgt_update += 1

    # Training
    def train(self, obs, act, tgt, size):

        self.q_val.train(obs, act, tgt, size)
        self.update_target()

    # Reset
    def reset(self):

        self.q_val.reset()
        self.q_tgt.reset()
