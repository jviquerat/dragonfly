# Generic imports
import numpy as np

# Custom imports
from dragonfly.core.network     import *
from dragonfly.core.optimizer   import *
from dragonfly.policies.factory import *

###############################################
### Actor class
### act_dim : output dimension
### obs_dim : input  dimension
### pms     : parameters
class actor():
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.policy  = policy_factory.create(pms.pol_type,
                                             act_dim = act_dim)

        # Define and init network
        self.net = network(obs_dim, self.policy.dim, pms)

        # Define optimizer
        self.opt = optimizer(pms.lr, pms.grd_clip,
                             self.net.trainable_weights)

    # Network forward pass
    def call(self, state):

        return self.net.call(state)

    # Get actions
    def get_action(self, obs):

        # Cast
        obs = tf.cast([obs], tf.float32)

        # Forward pass to get policy parameters
        policy_params = self.call(obs)
        action        = self.policy.call(policy_params)

        return action

    # Save network weights
    def save_weights(self):

        self.weights = self.net.get_weights()

    # Set network weights
    def set_weights(self, weights):

        self.net.set_weights(weights)

    # Get current learning rate
    def get_lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()
