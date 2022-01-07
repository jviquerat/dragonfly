# Generic imports
import numpy                as np
from   copy import deepcopy as cp

# Custom imports
from dragonfly.src.policy.tfd          import *
from dragonfly.src.network.network     import *
from dragonfly.src.optimizer.optimizer import *
from dragonfly.src.loss.loss           import *

###############################################
### Base policy
class base_policy():
    def __init__(self):
        pass

    # Get actions
    def get_actions(self, obs):
        raise NotImplementedError

    # Compute pdf
    def compute_pdf(self, obs):
        raise NotImplementedError

    # Reshape actions for training
    def reshape_actions(self, act):
        raise NotImplementedError

    # Networks forward pass
    def call_net(self, state):
        raise NotImplementedError

    # Save network weights
    def save_weights(self):

        self.weights = self.net.get_weights()

    # Set network weights
    def set_weights(self, weights):

        self.net.set_weights(weights)

    # Call loss for training
    def train(self, obs, adv, act, lgp):

        return self.loss.train(obs, adv, act, lgp, self)

    # Get current learning rate
    def get_lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()
        self.pdf = None
