# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.srl.base            import *
from dragonfly.src.core.paths          import *
from dragonfly.src.network.network     import net_factory
from dragonfly.src.optimizer.optimizer import opt_factory
from dragonfly.src.loss.loss           import loss_factory

###############################################
### Convolutional autoencoder srl class
class cae(base_srl):
    def __init__(self, obs_dim, buff_size, pms):

        # Initialize from arguments
        self.name             = "cae"
        self.obs_dim          = obs_dim
        self.buff_size        = buff_size
        self.latent_dim       = pms.latent_dim
        self.warmup           = pms.warmup
        self.retrain_freq     = pms.retrain_freq
        self.n_update_max     = pms.n_update_max
        self.batch_size       = pms.batch_size
        self.n_epochs_warmup  = pms.n_epochs_warmup
        self.n_epochs_retrain = pms.n_epochs_retrain

        # Initialize network
        self.net = net_factory.create("conv2d_ae",
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

        log_length = self.n_epochs_warmup + (self.n_update_max-1)*self.n_epochs_retrain
        self.loss_log = np.zeros((log_length, 2))
        self.n_epoch  = 0
        self.counter  = 0
        self.n_update = 0

    # Update autoencoder
    def update(self):

        if (self.n_update == 0): n_epochs = self.n_epochs_warmup
        else: n_epochs = self.n_epochs_retrain

        # Update
        for i in range(n_epochs):
            obs  = self.gbuff.get_batches(["obs"], self.batch_size)["obs"]
            loss = self.loss.train(obs, self)
            self.loss_log[self.n_epoch,0] = self.n_epoch
            self.loss_log[self.n_epoch,1] = loss
            self.n_epoch += 1

        # Write loss to file
        filename = paths.run + '/cae_loss.dat'
        np.savetxt(filename, self.loss_log)

        # Save weights
        filename = paths.run + '/' + self.name
        self.save(filename)

    # Full network forward pass
    def forward(self, state):

        return self.net.call(state)[0]

    # Process raw observations
    def process(self, obs):

        return self.net.encoder(obs)[0].numpy()

    # Save network weights
    def save(self, filename):

        self.net.save_weights(filename + '.weights.h5')

    # Load network weights
    def load(self, filename):

        load_status = self.net.load_weights(filename + '.weights.h5')
