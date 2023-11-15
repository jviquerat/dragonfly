# Generic imports
import numpy as np
from numpy import linalg as la
from numpy import exp

# Custom imports
from dragonfly.src.srl.base import *

###############################################
### Class for Kernel PCA srl
### pms : parameters
class kpca():
    def __init__(self, dim, new_dim, freq, size):

        # Initialize from arguments
        self.obs_dim = dim
        self.buff_size = size
        self.reduced_dim = new_dim
        self.freq = freq

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
        obs = obs.numpy()[-self.freq:]
        self.obs = obs
        
        # Pairwise squared Euclidean distances
        K = GramMat(obs,obs)
        
        # Compute centered symmetric kernel matrix
        K = kernel(K)
        K = center(K)
        
        # PCA algorithm
        evals, evecs = la.eigh(K)
        idx = np.argsort(evals)[::-1]
        self.evecs = evecs[:,idx]
        self.evals = evals[idx]
        	
    # Process observations
    def process(self, obs):
        
        # Before the update time
        if self.counter < self.freq :
            return obs[:,:self.reduced_dim]

        # Check if it's the update time
        if (self.counter == self.freq) :
            self.update()

        # Compute centered gram matrix between old obs and new obs
        K = GramMat(obs, self.obs)
        K = kernel(K)
        K = center(K)
        
        # Project obs into new space
        scales = np.sqrt(self.evals[:self.reduced_dim])
        scaled_alpha = self.evecs[:,:self.reduced_dim]/scales
        return np.dot(K, scaled_alpha)


# Gram Matrix of squared Euclidean distances between X and Y
"""
def GramMat(X,Y):
    n = X.shape[0]
    m = Y.shape[0]
    M = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            M[i,j] = sum([a*a for a in X[i,:]-Y[j,:]])
    return M
"""
def GramMat(X,Y):
    n,p = X.shape
    m,q = Y.shape
    assert(p==q)
    one_pm = np.ones((p,m))
    one_nq = np.ones((n,q))
    XX = (X*X) @ one_pm
    YY = one_nq @ (Y*Y).T
    XY = X @ (Y.T)
    return XX + YY - 2*XY 
        
# Center Gram Matrix
def center(K):
    n, m = K.shape
    one_n = np.ones((n,n))/n
    one_m = np.ones((m,m))/m
    K = K - one_n.dot(K) - K.dot(one_m) + one_n.dot(K).dot(one_m)
    return K
    
# RBF Kernel function for pairwise distance matrix
def kernel(M):
    gamma = 1
    K = exp(-gamma * M)
    return K
