# Generic imports
import math
import numpy as np

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
    def actions(self, obs):
        raise NotImplementedError

    # Control (deterministic actions)
    def control(self, obs):
        raise NotImplementedError

    # Compute pdf
    def compute_pdf(self, obs):
        raise NotImplementedError

    # Reshape actions for training
    def reshape_actions(self, act):
        raise NotImplementedError

    # Networks forward pass
    def forward(self, state):
        raise NotImplementedError

    # Save network weights
    def save_weights(self):

        self.weights = self.net.get_weights()

    # Set network weights
    def set_weights(self, weights):

        self.net.set_weights(weights)

    # Get current learning rate
    def lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()
        self.pdf = None

    # Save
    def save(self, filename):

        self.net.save_weights(filename)

    # Load
    def load(self, filename):

        load_status = self.net.load_weights(filename)
        load_status.assert_consumed()

    # Copy net into tgt
    def copy_tgt(self):

        self.tgt.set_weights(self.net.get_weights())
