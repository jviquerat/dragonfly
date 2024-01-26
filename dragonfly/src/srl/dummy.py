# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.srl.base import *

###############################################
### Class for dummy srl
class dummy(base_srl):
    def __init__(self, obs_dim, buff_size, pms):

        # Initialize from arguments
        self.obs_dim = obs_dim
        self.buff_size = buff_size
        self.counter = 0

    # Process observations
    def process(self, obs):

        return obs

    def store(self, name, x):
        pass
