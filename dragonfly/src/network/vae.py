# Custom imports
from dragonfly.src.network.base import *

###############################################
### Variational autoencoder network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class vae(base_network):
    def __init__(self, inp_dim, lat_dim, pms):

        # Initialize base class
        super().__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.lat_dim = lat_dim

        # Set default values
        self.arch = [64]
        #self.actv = "relu"
        self.actv = tf.nn.leaky_relu

        # Check inputs
        if hasattr(pms, "arch"):  self.arch = pms.arch
        #if hasattr(pms, "actv"):  self.actv = pms.actv

        # Initialize network
        self.net = []

        # Define encoder
        for l in range(len(self.arch)):
            self.net.append(Dense(self.arch[l],
                                  activation = self.actv))
        self.net.append(Dense(self.lat_dim, activation = "linear"))
        self.net.append(Dense(self.lat_dim, activation = "linear"))

        # Define decoder
        for l in range(len(self.arch)):
            self.net.append(Dense(self.arch[-l],
                                  activation = self.actv))
        self.net.append(Dense(self.inp_dim, activation = "linear"))

        # Initialize weights
        dummy = self.call(tf.ones([1,self.inp_dim]))

        # Save initial weights
        self.init_weights = self.get_weights()

    # Network forward pass
    @tf.function
    def call(self, var):

        # Initialize
        i   = 0
        out = []

        # Compute encoder
        for l in range(len(self.arch)):
            var = self.net[i](var)
            i  += 1

        # Output mean
        mean = self.net[i](var)
        i += 1

        # Output std
        #std = self.net[i](var)
        log_std = self.net[i](var)
        std     = tf.math.exp(log_std)
        i += 1

        # Sample
        lat = self.reparameterize(mean, std)
        var = lat

        # Compute decoder
        for l in range(len(self.arch)):
            var = self.net[i](var)
            i  += 1
        var = self.net[i](var)
        i += 1

        out.append(var)

        return out, mean, std

    # Reparameterization trick
    @tf.function
    def reparameterize(self, mean, std):

        epsilon = tf.random.normal(shape=(tf.shape(std)))
        z       = mean + std*epsilon

        return z


    # Network forward pass
    @tf.function
    def encoder(self, var):

        # Initialize
        i   = 0
        out = []

        # Compute encoder
        for l in range(len(self.arch)):
            var = self.net[i](var)
            i  += 1

        # Output mean
        mean = self.net[i](var)
        i += 1

        # Output std
        #std = self.net[i](var)
        log_std = self.net[i](var)
        std     = tf.math.exp(log_std)
        i += 1

        # Sample
        lat = self.reparameterize(mean, std)

        out.append(lat)

        return out

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
