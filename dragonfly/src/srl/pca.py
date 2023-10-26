# Generic imports
import numpy as np
from scipy import linalg as la

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
		self.counter = counter(1)
		
		# Initialize projection matrix
		self.projection = np.identity(self.obs_dim)
		
		# Create buffers
		self.names = ["obs"]
		self.sizes = [self.obs_dim]
		self.gbuff = gbuff(self.buff_size, self.names, self.sizes)
		
		
	# Update compression process according to the new buffer
	def update(self):	
	
		# Get data
		obs = self.gbuff["obs"]
		# PCA algorithm	
		obs -= obs.mean(axis=0)
		R = np.cov(obs,rowvar=False)	    
		evals, evecs = la.eigh(R)
		idx = np.argsort(evals)[::-1]
		evecs = evecs[:,idx]
		# Update projection matrix	
		self.projection = evecs[:, :self.reduced_dim]
		
	# Process observations
	def process(self, obs):
	
	   	# Check if it's the update time
	   	if self.counter.step == freq_srl :
	   		self.update()       
	   		self.counter.reset()
	   		
	   	# Project obs into new space
	   	Mult = np.matmul(self.projection,obs.T)								
	   	return Mult.T

