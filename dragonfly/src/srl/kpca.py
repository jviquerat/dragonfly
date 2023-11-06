# Generic imports
import numpy as np
from scipy import linalg as la
from scipy.spatial.distance import pdist, squareform
from scipy import exp

# Custom imports
from dragonfly.src.srl.base import *

###############################################
### Class for PCA srl
### pms : parameters
class kpca():
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
        
        # Pairwise squared Euclidean distances
        Dists = squareform(pdist(obs,'sqeuclidean'))
        
        # Symmetric RBF kernel matrix
        gamma = 10
        self.K = exp(-gamma * Dists)
        
        # Center the kernel matrix
        n = self.K.shape[0]
        one_n = np.ones((n,n))/n
        self.K -= one_n.dot(self.K) - self.K.dot(one_n) + one_n.dot(self.K).dot(one_n)
        
        # PCA algorithm
        evals, evecs = la.eigh(self.K)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        # Update projection matrix
        self.matrix = evecs
	
    # Process observations
    def process(self, obs):
        
        if self.counter < freq_srl :
            self.K = obs

        # Check if it's the update time
        if (self.counter % freq_srl) == 0 :
            self.update()

        # Reduce the dimension
        self.projection = self.matrix[:,:self.reduced_dim]        
        # Project obs into new space
        Mult = np.matmul(self.K,self.projection)
        return Mult
    
