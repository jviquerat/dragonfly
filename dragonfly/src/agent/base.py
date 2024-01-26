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
    def init_srl(self, pms, obs_dim, buff_size):

        # Check inputs
        if not hasattr(pms, "srl"):
            pms.srl.type = "dummy"

        # Create srl
        self.srl = srl_factory.create(pms.srl.type,
                                      obs_dim   = obs_dim,
                                      buff_size = buff_size,
                                      pms       = pms.srl)

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
