# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.srl.base import *

###############################################
### Class for dummy srl
### pms : parameters
class dummy():
    def __init__(self, dim, size):
    	
    	# Initialize from arguments
    	self.obs_dim = dim 
    	self.buff_size = size
    	
    	# Initialize counter
    	self.counter = counter(1)
    	
    	# Create buffers
    	self.names = ["obs"]
    	self.sizes = [self.obs_dim]
    	self.gbuff = gbuff(self.buff_size, self.names, self.sizes)

    # Process observations
    def process(self, obs):

        return obs
