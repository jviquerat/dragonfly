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

    # Compute previous pdf
    def compute_prp(self, obs):
        raise NotImplementedError

    # Reshape actions for training
    def reshape_actions(self, act):
        raise NotImplementedError

    # Networks forward pass
    def call_net(self, state):
        raise NotImplementedError

    # Previous networks forward pass
    def call_prn(self, state):
        raise NotImplementedError

    # Call loss for training
    def train(self, obs, adv, act):

        return self.loss.train(obs, adv, act, self)

    # Save previous policy
    def save_prv(self):

        self.prn.set_weights(self.net.get_weights())
        self.prp = self.pdf.copy()

    # Get current learning rate
    def get_lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()
        self.pdf = None

        if (self.save):
            self.prn.reset()
            self.prp = None
