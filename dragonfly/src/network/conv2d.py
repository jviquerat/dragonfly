# Generic imports
from tensorflow.keras.layers    import Conv2D, MaxPool2D

# Custom imports
from dragonfly.src.network.base import *

###############################################
### CNN class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class conv2d(base_network):
    def __init__(self, inp_dim, inp_shape, out_dim, pms):

        # Initialize base class
        super().__init__()

        # Set inputs
        self.inp_dim   = inp_dim
        self.inp_shape = inp_shape
        self.out_dim   = out_dim

        # Set default values
        self.trunk         = trunk()
        self.trunk.filters = [64]
        self.trunk.kernels = [3]
        self.trunk.stride  = 1
        self.trunk.actv    = "relu"
        self.heads         = heads()
        self.heads.nb      = 1
        self.heads.arch    = [[64]]
        self.heads.actv    = ["relu"]
        self.heads.final   = ["linear"]
        self.k_init        = Orthogonal(gain=1.0)
        self.k_init_final  = Orthogonal(gain=0.0)
        self.pooling       = False

        # Check inputs
        if hasattr(pms,       "trunk"):        self.trunk         = pms.trunk
        if hasattr(pms.trunk, "filters"):      self.trunk.filters = pms.trunk.filters
        if hasattr(pms.trunk, "kernels"):      self.trunk.kernels  = pms.trunk.kernels
        if hasattr(pms.trunk, "strides"):      self.trunk.stride  = pms.trunk.stride
        if hasattr(pms.trunk, "actv"):         self.trunk.actv    = pms.trunk.actv
        if hasattr(pms,       "heads"):        self.heads         = pms.heads
        if hasattr(pms.heads, "nb"):           self.heads.nb      = pms.heads.nb
        if hasattr(pms.heads, "arch"):         self.heads.arch    = pms.heads.arch
        if hasattr(pms.heads, "actv"):         self.heads.actv    = pms.heads.actv
        if hasattr(pms.heads, "final"):        self.heads.final   = pms.heads.final
        if hasattr(pms,       "k_init"):       self.k_init        = pms.k_init
        if hasattr(pms,       "k_init_final"): self.k_init_final  = pms.k_init_final
        if hasattr(pms,       "pooling"):      self.pooling       = pms.pooling

        # Check that out_dim and heads have same dimension
        if (len(self.out_dim) != pms.heads.nb):
            error("conv2d", "__init__",
                  "Out_dim and heads should have same dimension")

        # Initialize network
        self.net = []

        assert len(self.trunk.filters) == len(
            self.trunk.kernels
        ), f"Wrong kernel list. Expected {len(self.trunk.filters)} elements, got {len(self.trunk.kernels)}."

        # Define trunk
        for l in range(len(self.trunk.filters)):
            if (l == 0):
                self.net.append(Conv2D(filters            = self.trunk.filters[l],
                                       kernel_size        = self.trunk.kernels[l],
                                       strides            = self.trunk.stride,
                                       kernel_initializer = self.k_init,
                                       activation         = self.trunk.actv,
                                       input_shape        = self.inp_shape,
                                       padding            = "same"))
            else:
                self.net.append(Conv2D(filters            = self.trunk.filters[l],
                                       kernel_size        = self.trunk.kernels[l],
                                       strides            = self.trunk.stride,
                                       kernel_initializer = self.k_init,
                                       activation         = self.trunk.actv,
                                       padding            = "same"))

            if (self.pooling):
                self.net.append(MaxPool2D(pool_size = 2))

        # Define heads
        for h in range(self.heads.nb):
            for l in range(len(self.heads.arch[h])):
                self.net.append(Dense(self.heads.arch[h][l],
                                      kernel_initializer = self.k_init,
                                      activation         = self.heads.actv[h]))
            self.net.append(Dense(self.out_dim[h],
                                  kernel_initializer = self.k_init_final,
                                  activation         = self.heads.final[h]))

        # Initialize weights
        dummy = self.call(tf.ones([1] + self.inp_shape))

        # Save initial weights
        self.init_weights = self.get_weights()

    # Network forward pass
    @tf.function
    def call(self, var):

        # Initialize
        i   = 0
        out = []

        # Back to the original dimension
        var = Reshape(self.inp_shape)(var)

        # Compute trunk
        for l in range(len(self.trunk.filters)):
            var = self.net[i](var)
            i  += 1

            if (self.pooling):
                var = self.net[i](var)
                i  += 1

        var = Flatten()(var)

        # Compute heads
        # Each head output is stored in a tf.tensor, and
        # is appended to the global output
        for h in range(self.heads.nb):
            hvar = var
            for l in range(len(self.heads.arch[h])):
                hvar = self.net[i](hvar)
                i   += 1
            hvar = self.net[i](hvar)
            i   += 1
            out.append(hvar)

        return out

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
