# Generic imports
import numpy as np

# Custom imports
from dragonfly.core.network     import *
from dragonfly.core.optimizer   import *

###############################################
### Multinomial policy class (discrete)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class multinomial():
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.dim     = self.act_dim

        # Define and init network
        self.net = network(obs_dim, self.dim, pms.network)

        # Define optimizer
        self.opt = optimizer(pms.optimizer,
                             self.net.trainable_weights)

    # Get action
    def get_action(self, obs):

        # Cast
        obs = tf.cast([obs], tf.float32)

        # Forward pass to get policy parameters
        policy_params = self.call(obs)

        # Sanitize output
        policy       = tf.cast(policy_params, tf.float64)
        policy, norm = tf.linalg.normalize(policy, ord=1)

        policy       = np.asarray(policy)[0]
        action       = np.random.multinomial(1, policy)
        action       = np.float32(action)

        return action

    # Network forward pass
    def call(self, state):

        return self.net.call(state)

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
