# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.srl.base import *

###############################################
### Class for PCA srl
### pms : parameters
class pca(base_srl):
    def __init__(self, obs_dim, buff_size, pms):

        # Init base class
        super().__init__()

        # Initialize from arguments
        self.obs_dim       = obs_dim
        self.buff_size     = buff_size
        self.latent_dim    = pms.latent_dim
        self.update_freq   = pms.update_freq

        # Possible single update of PCA kernel
        self.single_update = False
        self.updated = False
        if hasattr(pms, "single_update"):
                self.single_update = pms.single_update

        # Running mean and std for latent observations
        self.mean = np.zeros(self.latent_dim)
        self.std  = np.zeros(self.latent_dim)

        # Initialize projection matrix
        self.matrix = np.zeros((self.obs_dim, self.latent_dim))
        for i in range(self.latent_dim):
            self.matrix[i,i] = 1

        # Create buffers
        self.names = ["obs"]
        self.sizes = [self.obs_dim]
        self.gbuff = gbuff(self.buff_size, self.names, self.sizes)

    # Update compression process according to the new buffer
    def update(self):

        # If single update
        if (self.single_update and self.updated): return

        print("UPDATE PCA")

        # Get data
        obs = self.gbuff.get_buffers(["obs"], self.gbuff.length())["obs"]
        obs = obs.numpy()

        # Normalize data
        mu  = obs.mean(axis=0)
        std = obs.std(axis=0)
        #obs = (obs - mu)/(std + 1.0e-8)
        obs = obs - mu

        # obs -= obs.mean(axis=0)
        # std = obs.std(axis=0)
        # index = np.where(std!=0)[0]
        # obs[:,index] /= std[index]

        # PCA algorithm
        m = np.cov(obs, rowvar=False)


        # Compute svd
        vects, vals, v = np.linalg.svd(m)

        # Re-order based on singular value magnitudes
        order        = np.flip(np.argsort(vals))
        reduced_vals = vals[order[:self.latent_dim]]
        self.matrix  = vects[:,order[:self.latent_dim]]

        # Compute explained variance
        explv = np.sum(reduced_vals)/np.sum(vals)

        #evals, evecs = np.linalg.eigh(R)
        #idx          = np.argsort(evals)[::-1]
        #evecs        = evecs[:,idx]

        # Update projection matrix
        #self.matrix = evecs

        #alpha = 0.9
        #lobs = np.matmul(obs, self.matrix[:,:self.latent_dim])
        #self.mean = alpha*self.mean  + (1.0-alpha)*np.mean(lobs, axis=0)
        #self.std = alpha*self.std + (1.0-alpha)*np.std(lobs, axis=0)

        self.updated = True

    # Process observation
    def process(self, obs):

        # Check if it's the update time
        if ((self.gbuff.length() > 0) and (self.counter > self.update_freq)):
            self.update()
            self.counter = 0

        # Reduce the dimension
        #self.latant_ = self.matrix[:,:self.latent_dim]

        # Project obs into new space
        x          = np.reshape(obs, (len(obs), -1))
        latent_obs = np.matmul(x, self.matrix)

        #print(obs)
        #print(latent_obs)
        #print("")

        #Mult = (Mult - self.mean)/(self.std + 1.0e-8)

        return latent_obs



