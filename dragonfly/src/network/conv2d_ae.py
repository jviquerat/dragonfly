# Generic imports
from tensorflow.keras.layers    import Conv2D, MaxPool2D, Conv2DTranspose 

# Custom imports
from dragonfly.src.network.base import *

###############################################
### Autoencoder network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class conv2d_ae(base_network):
    def __init__(self, inp_dim, lat_dim, pms):

        # Initialize base class
        super().__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.lat_dim = lat_dim

        # Set default values
        self.conv          = trunk()
        self.conv.filters  = [64]
        self.conv.kernel   = 3
        self.conv.stride   = 1
        self.conv.actv     = "relu"
        self.fc            = heads()
        self.fc.arch       = [64]
        self.fc.actv       = "relu"
        self.k_init        = Orthogonal(gain=1.0)
        self.original_dim  = None
        self.pooling       = False

        # Check inputs
        if hasattr(pms,      "trunk"):         self.conv          = pms.conv
        if hasattr(pms.conv, "filters"):       self.conv.filters  = pms.conv.filters
        if hasattr(pms.conv, "k_size"):        self.conv.kernel   = pms.conv.kernel
        if hasattr(pms.conv, "strides"):       self.conv.stride   = pms.conv.stride
        if hasattr(pms.conv, "actv"):          self.conv.actv     = pms.conv.actv
        if hasattr(pms,      "heads"):         self.fc            = pms.fc
        if hasattr(pms.fc,   "arch"):          self.fc.arch       = pms.fc.arch
        if hasattr(pms.fc,   "actv"):          self.fc.actv       = pms.fc.actv
        if hasattr(pms,      "k_init"):        self.k_init        = pms.k_init
        if hasattr(pms,      "original_dim"):  self.original_dim  = pms.original_dim
        if hasattr(pms,      "pooling"):       self.pooling       = pms.pooling

        # Specific dimensions
        self.nx       = self.original_dim[0]
        self.ny       = self.original_dim[1]
        self.channels = self.original_dim[2]
            

        # Initialize network
        self.net = []

        # Define encoder
        # Encoder: convolution part
        for l in range(len(self.conv.filters)):
            if (l == 0):
                self.net.append(Conv2D(filters            = self.conv.filters[l],
                                       kernel_size        = self.conv.kernel,
                                       strides            = self.conv.stride,
                                       kernel_initializer = self.k_init,
                                       activation         = self.conv.actv,
                                       input_shape        = self.original_dim,
                                       padding            = "same"))
            else:
                self.net.append(Conv2D(filters            = self.conv.filters[l],
                                       kernel_size        = self.conv.kernel,
                                       strides            = self.conv.stride,
                                       kernel_initializer = self.k_init,
                                       activation         = self.conv.actv,
                                       padding            = "same"))

            if (self.pooling):
                self.net.append(MaxPool2D(pool_size = 2))

        # Encoder: fully connected part
        for l in range(len(self.fc.arch)):
            self.net.append(Dense(self.fc.arch[l],
                                  activation = self.fc.actv))
            
        # Define bottleneck
        self.net.append(Dense(self.lat_dim, activation = "linear"))

        # Define decoder
        # Decoder: fully connected part
        for l in range(1,len(self.fc.arch)+1):
            self.net.append(Dense(self.fc.arch[-l],
                                  activation = self.fc.actv))

        # Decoder convolution part
        for l in range(1,len(self.conv.filters)):
            self.net.append(Conv2DTranspose(filters     = self.conv.filters[-l],
                                            kernel_size = self.conv.kernel,
                                            strides     = self.conv.stride,
                                            activation  = self.conv.actv,
                                            padding     = "same"))
            
        self.net.append(Conv2DTranspose(filters         = self.channels,
                                            kernel_size = self.conv.kernel,
                                            strides     = self.conv.stride,
                                            activation  = self.conv.actv,
                                            padding     = "same"))    

        # Initialize weights
        dummy = self.call(tf.ones([1]+self.original_dim))

        # Save initial weights
        self.init_weights = self.get_weights()

    # Network forward pass
    @tf.function
    def call(self, var):

        # Initialize
        i   = 0
        out = []

        # Back to the original dimension
        var = Reshape(self.original_dim)(var)

        # Compute encoder
        # Encoder: convolution part
        for l in range(len(self.conv.filters)):
            var = self.net[i](var)
            i  += 1

            if (self.pooling):
                var = self.net[i](var)
                i  += 1

        conv_final_shape = var.shape[1:]        
        var = Flatten()(var)
        flat_final_shape = var.shape[1]
        
        # Encoder: fully connected part
        for l in range(len(self.fc.arch)):
            var = self.net[i](var)
            i  += 1

        # Compute bottleneck
        var = self.net[i](var)
        i  += 1

        # Compute decoder
        # Decoder: fully connected part
        for l in range(len(self.fc.arch)):
            var = self.net[i](var)
            i  += 1

        # Join to convolution
        var = Reshape(conv_final_shape)(var)

        # Decoder: convolution part
        for l in range(len(self.conv.filters)):
            var = self.net[i](var)
            i  += 1

        var = Flatten()(var)
        out.append(var)

        return out

    # Network forward pass
    @tf.function
    def encoder(self, var):

        # Initialize
        i   = 0
        out = []

        # Back to the original dimension
        var = Reshape(self.original_dim)(var)

        # Compute encoder
        # Encoder: convolution part
        for l in range(len(self.conv.filters)):
            var = self.net[i](var)
            i  += 1

            if (self.pooling):
                var = self.net[i](var)
                i  += 1

        var = Flatten()(var)

        # Encoder: fully connected part
        for l in range(len(self.fc.arch)):
            var = self.net[i](var)
            i  += 1

        # Compute bottleneck
        lat = self.net[i](var)

        out.append(lat)

        return out

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
