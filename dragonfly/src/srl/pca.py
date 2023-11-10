# Generic imports
import numpy as np
from numpy import linalg as la

# Custom imports
from dragonfly.src.srl.base import *

###############################################
### Class for PCA srl
### pms : parameters
class pca():
	def __init__(self, dim, size):
		
		# Initialize from arguments
		self.obs_dim = dim
		self.buff_size = size
		
		# Initialize counter
		self.counter = 1
		
		# Initialize projection matrix
		self.matrix = np.identity(self.obs_dim)
		
		# Create buffers
		self.names = ["obs"]
		self.sizes = [self.obs_dim]
		self.gbuff = gbuff(self.buff_size, self.names, self.sizes)
                		
	# Update compression process according to the new buffer
	def update(self):

                # Get data
                obs = self.gbuff.get_buffers({"obs"},self.counter)["obs"]
                obs = obs.numpy()
                # Normalize data
                obs -= obs.mean(axis=0)
                std = obs.std(axis=0)
                index = np.where(std!=0)[0]
                obs[:,index] /= std[index]
                # PCA algorithm
                R = np.cov(obs,rowvar=False)
                evals, evecs = la.eigh(R)
                idx = np.argsort(evals)[::-1]
                evecs = evecs[:,idx]
                # Update projection matrix
                self.matrix = evecs
		
	# Process observations
	def process(self, obs):
                
                # Check if it's the update time
                if ((self.counter-20000) % self.freq) == 0 :
                        self.update()
                        
                # Reduce the dimension
                self.projection = self.matrix[:,:self.reduced_dim]        
                # Project obs into new space
                Mult = np.matmul(obs,self.projection)
                return Mult



