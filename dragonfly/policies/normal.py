# Generic imports
import numpy      as np

# Tensorflow imports
import tensorflow as tf

###############################################
### Normal policy class (continuous)
### act_dim  : size of action vector required by environment
class normal():
    def __init__(self, act_dim):

        # Dimension of policy output
        self.dim  = 2*act_dim

    # Call policy
    def call(self, params):

        print("Normal law policy not implemented yet")
        exit()
