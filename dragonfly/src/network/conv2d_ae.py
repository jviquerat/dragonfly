# Generic imports
import math
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
        self.trunk         = trunk()
        self.trunk.filters = [64]
        self.trunk.kernels = [3]
        self.trunk.stride  = 1
        self.trunk.actv    = "relu"
        self.k_init        = Orthogonal(gain=1.0)
        self.original_dim  = None
        self.pooling       = False

        # Check inputs
        if hasattr(pms,       "trunk"):         self.trunk         = pms.trunk
        if hasattr(pms.trunk, "filters"):       self.trunk.filters = pms.trunk.filters
        if hasattr(pms.trunk, "kernels"):       self.trunk.kernels = pms.trunk.kernels
        if hasattr(pms.trunk, "strides"):       self.trunk.stride  = pms.trunk.stride
        if hasattr(pms.trunk, "actv"):          self.trunk.actv    = pms.trunk.actv
        if hasattr(pms,       "k_init"):        self.k_init        = pms.k_init
        if hasattr(pms,       "original_dim"):  self.original_dim  = pms.original_dim
        if hasattr(pms,       "pooling"):       self.pooling       = pms.pooling

        # Specific dimensions
        self.nx    = self.original_dim[0]
        self.ny    = self.original_dim[1]
        self.stack = self.original_dim[2]

        # Initialize network
        self.enc = []
        self.dec = []
        self.w   = self.nx
        self.h   = self.ny

        # Define encoder
        for l in range(len(self.trunk.filters)):
            if (l == 0):
                self.enc.append(Conv2D(filters            = self.trunk.filters[l],
                                       kernel_size        = self.trunk.kernels[l],
                                       strides            = self.trunk.stride,
                                       kernel_initializer = self.k_init,
                                       activation         = self.trunk.actv,
                                       input_shape        = self.original_dim,
                                       padding            = "same"))
            else:
                self.enc.append(Conv2D(filters            = self.trunk.filters[l],
                                       kernel_size        = self.trunk.kernels[l],
                                       strides            = self.trunk.stride,
                                       kernel_initializer = self.k_init,
                                       activation         = self.trunk.actv,
                                       padding            = "same"))
            self.w = self.compute_shape(self.w, self.trunk.kernels[l], 2, self.trunk.stride)
            self.h = self.compute_shape(self.h, self.trunk.kernels[l], 2, self.trunk.stride)

            if (self.pooling):
                self.enc.append(MaxPool2D(pool_size = 2))
                self.w = self.compute_shape(self.w, 2, 0, 1)
                self.h = self.compute_shape(self.h, 2, 0, 1)


        # Linear layer to bottleneck
        self.enc.append(Dense(self.lat_dim, activation = "linear"))

        # Linear layer from bottleneck
        d = self.w*self.h*self.trunk.filters[-1]
        self.dec.append(Dense(d, activation = "linear"))

        # Define decoder
        for l in range(len(self.trunk.filters)):
            self.dec.append(Conv2DTranspose(filters     = self.trunk.filters[-l-1],
                                            kernel_size = self.trunk.kernels[-l-1],
                                            strides     = self.trunk.stride,
                                            activation  = self.trunk.actv,
                                            padding     = "same"))

        self.dec.append(Conv2D(filters     = self.stack,
                               kernel_size = self.trunk.kernels[0],
                               strides     = self.trunk.stride,
                               activation  = "sigmoid",
                               padding     = "same"))

        # Initialize weights
        dummy = self.call(tf.ones([1, self.nx, self.ny, self.stack]))

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

        # Back to the original dimension
        # Reminder : the new shape will be (batch_size, self.original_dim)
        var = Reshape(self.original_dim)(var)

        # Convolutional layers
        for l in range(len(self.trunk.filters)):
            var = self.enc[i](var)
            i  += 1

            if (self.pooling):
                var = self.enc[i](var)
                i  += 1

        # Linear layer to bottleneck
        var        = Flatten()(var)
        var        = self.enc[i](var)
        i         += 1

        return [var]

    # Decoder forward pass
    @tf.function
    def decoder(self, var):

        # Initialize
        i = 0

        # Linear layer from bottleneck
        var        = self.dec[i](var)
        i         += 1
        var        = Reshape([self.w, self.h, self.trunk.filters[-1]])(var)

        # Transposed convolutional layers
        for l in range(len(self.trunk.filters)):
            var = self.dec[i](var)
            i  += 1

        # Last conv layer
        var = self.dec[i](var)
        i  += 1

        var = Flatten()(var)

        return [var]


    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)

    # Compute output shape for conv layers
    # w = width (and height, assuming square images)
    # k = kernel size
    # p = padding
    # s = stride
    # d = dilation
    def compute_shape(self, w, k=3, p=0, s=1, d=1):

        return math.floor((w + 2*p - d*(k - 1) - 1)/s + 1)
