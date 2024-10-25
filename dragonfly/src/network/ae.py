# Custom imports
from dragonfly.src.network.base import *

###############################################
### Autoencoder network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class ae(base_network):
    def __init__(self, inp_dim, inp_shape, lat_dim, pms):

        # Initialize base class
        super().__init__()

        # Set inputs
        self.inp_dim   = inp_dim
        self.inp_shape = inp_shape
        self.lat_dim   = lat_dim

        # Set default values
        self.arch = [64]
        self.actv = "relu"

        # Check inputs
        if hasattr(pms, "arch"):  self.arch = pms.arch
        if hasattr(pms, "actv"):  self.actv = pms.actv

        self.actv = tf.nn.leaky_relu

        # Initialize network
        self.enc = []
        self.dec = []

        # Define encoder
        for l in range(len(self.arch)):
            self.enc.append(Dense(self.arch[l],
                                  activation = self.actv))
        self.enc.append(Dense(self.lat_dim, activation = "linear"))

        # Define decoder
        for l in range(len(self.arch)):
            self.dec.append(Dense(self.arch[-l-1],
                                  activation = self.actv))
        self.dec.append(Dense(self.inp_dim, activation = "linear"))

        # Initialize weights
        dummy = self.call(tf.ones([1,self.inp_dim]))

        # Save initial weights
        self.init_weights = self.get_weights()

    # Network forward pass
    @tf.function
    def call(self, var):

        v = self.encoder(var)[0]
        v = self.decoder(v)

        return v

    # Encoder forward pass
    @tf.function
    def encoder(self, var):

        # Initialize
        i = 0

        # Compute encoder
        for l in range(len(self.arch)):
            var = self.enc[i](var)
            i  += 1
        var = self.enc[i](var)
        i += 1

        return [var]

    # Decoder forward pass
    @tf.function
    def decoder(self, var):

        # Initialize
        i = 0

        # Compute decoder
        for l in range(len(self.arch)):
            var = self.dec[i](var)
            i  += 1
        var = self.dec[i](var)
        i += 1

        return [var]

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
