# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.srl.base   import *
from dragonfly.src.core.paths import *

###############################################
### Class for PCA srl
### pms : parameters
class pca(base_srl):
    def __init__(self, obs_dim, buff_size, pms):

        # Initialize from arguments
        self.name          = "pca"
        self.obs_dim       = obs_dim
        self.buff_size     = buff_size
        self.latent_dim    = pms.latent_dim
        self.warmup        = pms.warmup
        self.retrain_freq  = pms.retrain_freq
        self.n_update_max  = pms.n_update_max

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

    # Update pca
    def update(self):

        # Get data
        obs = self.gbuff.get_buffers(["obs"], self.gbuff.length())["obs"]
        obs = obs.numpy()

        # PCA algorithm
        m              = np.cov(obs, rowvar=False)
        vects, vals, v = np.linalg.svd(m)
        order          = np.flip(np.argsort(vals))
        self.vals      = vals[order[:self.latent_dim]]
        self.matrix    = vects[:,order[:self.latent_dim]]

        # Compute explained variance
        self.explv = np.sum(self.vals)/np.sum(vals)

        # Save parameters
        filename = paths.run + '/' + self.name
        self.save(filename)

    # Process raw observations
    def process(self, obs):

        # Project obs into latent space
        x          = np.reshape(obs, (len(obs), -1))
        latent_obs = np.matmul(x, self.matrix)

        return latent_obs

    # Save pca
    def save(self, filename):

        with open(filename, "w") as f:
            f.write(f"{self.latent_dim} \n")
            f.write(f"{self.obs_dim} \n")
            f.write(f"{self.buff_size} \n")
            f.write(f"{self.explv} \n")
            np.savetxt(f, self.matrix)

    # Load pca
    def load(self, filename):

         with open(filename, "r") as f:
            self.latent_dim = int(f.readline().split(" ")[0])
            self.obs_dim    = int(f.readline().split(" ")[0])
            self.buff_size  = int(f.readline().split(" ")[0])
            self.explv      = float(f.readline().split(" ")[0])
            self.matrix     = np.loadtxt(filename,
                                         skiprows=4,
                                         max_rows=self.obs_dim)
