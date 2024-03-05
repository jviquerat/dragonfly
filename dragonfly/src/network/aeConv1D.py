# Custom imports
from dragonfly.src.network.base import *

###############################################
### Autoencoder network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class aeConv1D(base_network):
    def __init__(self, inp_dim, lat_dim, pms):

        # Initialize base class
        super().__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.lat_dim = lat_dim

        # Set default values
        self.arch = [64]
        self.actv = "relu"

        # Check inputs
        if hasattr(pms, "arch"):  self.arch = pms.arch
        if hasattr(pms, "actv"):  self.actv = pms.actv

        self.actv = tf.nn.leaky_relu

        # Initialize network
        self.net = []

        # Define encoder
        for l in range(len(self.arch)):
            self.net.append(Conv1D(filters     = self.arch[l],
                                   kernel_size = 3,
                                   strides     = 2,
                                   padding     = "same",
                                   input_shape = [self.inp_dim,1],
                                   activation  = self.actv))
            self.net.append(MaxPooling1D(2))

        self.net.append(Flatten())
        self.net.append(Dense(self.lat_dim, activation = "linear"))

        # Define decoder
        for l in range(len(self.arch)):
            self.net.append(Conv1D(filters    = self.arch[-l],
                                   kernel_size = 3,
                                   strides     = 2,
                                   padding     = "same",
                                   input_shape = [self.lat_dim,1],
                                   activation  = self.actv))
            self.net.append(UpSampling1D(2))
            
        self.net.append(Flatten())
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

        var = Reshape([self.inp_dim,1])(var)
        
        # Compute encoder
        for l in range(len(self.arch)):
            var = self.net[i](var)
            i  += 1
            var = self.net[i](var)
            i  += 1
        var = self.net[i](var)
        i += 1
        var = self.net[i](var)
        i += 1
                        
        var = Reshape([self.lat_dim,1])(var)
                
        # Compute decoder
        for l in range(len(self.arch)):
            var = self.net[i](var)
            i  += 1
            var = self.net[i](var)
            i  += 1
        var = self.net[i](var)
        i += 1
        var = self.net[i](var)

        var = Reshape([self.inp_dim])(var)

        out.append(var)

        return out

    # Network forward pass
    @tf.function
    def encoder(self, var):

        # Initialize
        i   = 0
        out = []

        var = Reshape([self.inp_dim,1])(var)

        # Compute encoder
        for l in range(len(self.arch)):
            var = self.net[i](var)
            i  += 1
            var = self.net[i](var)
            i  += 1

        var = self.net[i](var)
        i += 1
        lat = self.net[i](var)
        lat = Reshape([self.lat_dim])(lat)
        
        out.append(lat)

        return out

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
