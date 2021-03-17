# Generic imports
import numpy as np

# Custom imports
from dragonfly.core.network     import *
from dragonfly.core.optimizer   import *

###############################################
### Normal policy class (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal():
    def __init__(self, obs_dim, act_dim, pms):

        # Dimension of policy output
        self.dim  = 2*act_dim

    # Call policy
    def call(self, params):

        print("Normal law policy not implemented yet")
        exit()
