# Generic imports
import numpy as np

# Keras imports
from tensorflow.keras import Model

# Custom imports
from dragonfly.src.network.network     import *
from dragonfly.src.optimizer.optimizer import *
from dragonfly.src.loss.loss           import *

###############################################
### Base value
class base_value(Model):
    def __init__(self):
        super(base_value, self).__init__()

    # Get values
    def get_values(self, obs):
        raise NotImplementedError

    # Network forward pass
    def call_net(self, state):

        return self.net.call(state)

    # Get network weights
    def get_weights(self):

        return self.net.get_weights()

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
