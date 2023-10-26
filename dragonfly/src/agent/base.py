# Generic imports
import random
import numpy as np

# Custom imports
from dragonfly.src.policy.policy           import *
from dragonfly.src.value.value             import *
from dragonfly.src.decay.decay             import *
from dragonfly.src.retrn.retrn             import *
from dragonfly.src.srl.srl                 import *
from dragonfly.src.core.constants          import *
from dragonfly.src.termination.termination import *
from dragonfly.src.utils.buff              import *
from dragonfly.src.utils.timer             import *
from dragonfly.src.utils.error             import *

###############################################
### Base agent
class base_agent():
    def __init__(self, pms):
        pass

    # Get actions
    def actions(self, obs):
        raise NotImplementedError

    # Initialize srl
    def init_srl(self, pms, obs_dim, size):

        # Check inputs
        self.srl_type = "dummy"
        if hasattr(pms, "srl"): self.srl_type = pms.srl.type

        # Create srl
        self.srl = srl_factory.create(self.srl_type, dim=obs_dim,
        												size=size)
        
        # Get the new dimension
        if hasattr(pms, "srl"): 
        	self.reduced_dim = pms.srl.reduced_dim

    # Pre-process observations using srl
    def process_obs(self, obs):

        return self.srl.process(obs)

    # Reset
    def reset(self):
        raise NotImplementedError

    # Save
    def save(self, filename):
        raise NotImplementedError

    # Load
    def load(self, filename):
        raise NotImplementedError
