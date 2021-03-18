# Generic imports
import numpy as np

# Custom imports
from dragonfly.core.network     import *
from dragonfly.core.optimizer   import *

###############################################
### V-value class
### obs_dim : input  dimension
### pms     : parameters
class v_value():
    def __init__(self, obs_dim, pms):

        # Fill structure
        self.dim = 1
        self.obs_dim = obs_dim

        # Define and init network
        # Force linear activation, as this is v-value network
        pms.network.fnl_actv = "linear"
        self.net = network(obs_dim, self.dim, pms.network)

        # Define optimizer
        self.opt = optimizer(pms.optimizer,
                             self.net.trainable_weights)

    # Get values
    def get_values(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Predict values
        values = np.array(self.call(obs))

        return values

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
