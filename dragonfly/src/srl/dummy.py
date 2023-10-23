# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.core.constants import *

###############################################
### Class for dummy srl
### pms : parameters
class dummy():
    def __init__(self, dim):
        pass

    # Process observations
    def process(self, obs):

        return obs
