# Generic imports
import random
import numpy as np

# Custom imports
from dragonfly.src.policy.policy           import *
from dragonfly.src.value.value             import *
from dragonfly.src.decay.decay             import *
from dragonfly.src.retrn.retrn             import *
from dragonfly.src.core.constants          import *
from dragonfly.src.utils.error             import *
from dragonfly.src.termination.termination import *
from dragonfly.src.utils.buff              import *

###############################################
### Base agent
class base_agent():
    def __init__(self):
        pass

    # Get actions
    def actions(self, obs):
        raise NotImplementedError

    # Reset
    def reset(self):
        raise NotImplementedError

    # Save
    def save(self, filename):
        raise NotImplementedError

    # Load
    def load(self, filename):
        raise NotImplementedError
