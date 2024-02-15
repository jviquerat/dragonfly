# Generic imports
import numpy as np
from numpy import linalg as la
from numpy import exp

# Custom imports
from dragonfly.src.srl.base import *

###############################################
### Class for Kernel PCA srl
### pms : parameters
class kpca(base_srl):
    def __init__(self, obs_dim, buff_size, pms):

        # Initialize from arguments
        self.obs_dim     = obs_dim
        self.buff_size   = buff_size
        self.latent_dim  = pms.latent_dim
        self.update_freq = pms.update_freq
        self.n_updates   = pms.n_updates
        self.sigma       = pms.sigma
        self.gamma       = 1/(2*(self.sigma**2))

        # Initialize counter
        self.counter = 1

        # Initialize projection matrix
        self.matrix = np.zeros((self.obs_dim, self.latent_dim))
        for i in range(self.latent_dim):
            self.matrix[i,i] = 1

        # Create buffers
        self.names = ["obs"]
        self.sizes = [self.obs_dim]
        self.gbuff = gbuff(self.buff_size, self.names, self.sizes)
    
    # Reset
    def reset(self):

        self.gbuff.reset()

        self.counter  = 0
        self.n_update = 0

    # Update compression process according to the new buffer
    def update(self):
        
        if (self.n_update >= self.n_updates): return

        print("UPDATE KPCA")

        # Get data
        obs = self.gbuff.get_buffers({"obs"},self.gbuff.length()-1)["obs"]
        obs = obs.numpy()#[-self.freq:]
        self.obs = obs
                                
        # Pairwise squared Euclidean distances
        K = GramMat(obs,obs)
        
        # Compute centered symmetric kernel matrix
        K = self._kernel(K)
        K = center(K)
        
        # PCA algorithm
        evals, evecs = la.eigh(K)
        idx = np.argsort(evals)[::-1]
        self.evecs = evecs[:,idx]
        self.evals = evals[idx]
        scales = np.sqrt(self.evals[:self.latent_dim])
        self.matrix = self.evecs[:,:self.latent_dim]/scales

        self.n_update += 1

        obs = self.gbuff.get_batches(["obs"], 1000)["obs"]
        encoded = self.process(obs)
        self.plot2Dencoded(encoded)
        	
    # Process observations
    def process(self, obs):
        
        # Check if it's the update time
        if ((self.gbuff.length() > 0) and (self.counter > self.update_freq)):
            self.update()
            self.counter = 0

        # Compute centered gram matrix between old obs and new obs
        try:
            K = GramMat(obs, self.obs)
            K = self._kernel(K)
            K = center(K)
        except AttributeError:
            K = obs
        
        # Project obs into new space
        latent_obs = np.dot(K, self.matrix)

        # Project obs into new space using approximation
        # x          = np.reshape(obs, (len(obs), -1))
        # Phi_obs    = np.apply_along_axis(self._approxPhi, 1, x)
        # latent_obs = np.dot(Phi_obs, self.matrix)
        return latent_obs

    # RBF Kernel function for pairwise distance matrix
    def _kernel(self,M):
        K = exp(-self.gamma * M)
        return K

    # Approximate the RBF transformation funcion by Fourier random features
    def _approxPhi(self,x):
        d = int(self.matrix.shape[0]/2)
        n = x.shape[0]
        w = np.random.normal(0,2*self.gamma,(d,n))
        wx = np.dot(w,x)
        c = np.cos(wx).reshape((d,1))
        s = np.sin(wx).reshape((d,1))
        cs = np.concatenate((c,s),axis=1)
        return cs.reshape((1,2*d))
        #l = []
        #for i in range(d):
        #    w = np.random.normal(0,2*self.gamma,n)
        #    wx = np.dot(w,x)
        #    l += [np.cos(wx),np.sin(wx)]
        #return np.array(l)


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
    
