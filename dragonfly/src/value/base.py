# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.network.network     import *
from dragonfly.src.optimizer.optimizer import *
from dragonfly.src.loss.loss           import *

###############################################
### Base value
class base_value():
    def __init__(self):
        pass

    # Get values
    def get_values(self, obs):
        raise NotImplementedError

    # Network forward pass
    def call_net(self, state):

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
