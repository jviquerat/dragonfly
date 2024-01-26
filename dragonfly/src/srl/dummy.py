# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.srl.base import *

###############################################
### Class for dummy srl
class dummy(base_srl):
    def __init__(self, obs_dim, buff_size, pms):

        # Init base class
        super().__init__()

        # Initialize from arguments
        self.obs_dim    = obs_dim
        self.buff_size  = buff_size
        self.latent_dim = self.obs_dim

    # Process observations
    def process(self, obs):

        return obs

    def store_obs(self, obs):
        pass
