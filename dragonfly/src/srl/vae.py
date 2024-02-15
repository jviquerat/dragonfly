# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.srl.base            import *
from dragonfly.src.network.network     import *
from dragonfly.src.optimizer.optimizer import *
from dragonfly.src.loss.loss           import *

###############################################
### Variational autoencoder srl class
class vae(base_srl):
    def __init__(self, obs_dim, buff_size, pms):

        # Initialize from arguments
        self.obs_dim       = obs_dim
        self.buff_size     = buff_size
        self.latent_dim    = pms.latent_dim
        self.update_freq   = pms.update_freq
        self.batch_size    = pms.batch_size
        self.n_epochs      = pms.n_epochs
        self.n_updates     = pms.n_updates

        # Initialize network
        self.net = net_factory.create("vae",
                                      inp_dim = self.obs_dim,
                                      lat_dim = self.latent_dim,
                                      pms     = pms.network)
        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainables)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

        # Create buffers
        self.names = ["obs"]
        self.sizes = [self.obs_dim]
        self.gbuff = gbuff(self.buff_size, self.names, self.sizes)

        self.reset()

    # Reset
    def reset(self):

        self.gbuff.reset()
        self.net.reset()
        self.opt.reset()

        self.counter  = 0
        self.n_update = 0

    # Update vae
    def update(self):

        if (self.n_update >= self.n_updates): return

        print("UPDATE VAE")

        # Update
        for i in range(self.n_epochs):
            obs = self.gbuff.get_batches(["obs"], self.batch_size)["obs"]
            loss = self.loss.train(obs, self)

            # Write to file
            with open("vae_loss.dat", "a") as f:
                f.write(str(loss.numpy())+"\n")

        self.n_update += 1
        
        obs = self.gbuff.get_batches(["obs"], 1000)["obs"]
        encoded = self.net.encoder(obs)[0].numpy()
        self.plot2Dencoded(encoded)

        
    # Full network forward pass
    def forward(self, state):

        x, mean, log_var = self.net.call(state)
        x = x[0]

        return x, mean, log_var

    # Process raw observations
    def process(self, obs):

        # Check if it's the update time
        if ((self.gbuff.length() > 0) and (self.counter > self.update_freq)):
            self.update()
            self.counter = 0

        encoded = self.net.encoder(obs)[0].numpy()

        return encoded
