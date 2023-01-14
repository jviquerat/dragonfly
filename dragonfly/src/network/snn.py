# Tensorflow imports
from tensorflow.keras.layers    import AlphaDropout

# Custom imports
from dragonfly.src.network.base import *

###############################################
### Fully-connected network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class snn(base_network):
    def __init__(self, inp_dim, out_dim, pms):
        super().__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Set default values
        self.trunk          = trunk()
        self.trunk.arch     = [64]
        self.trunk.actv     = "selu"
        self.heads          = heads()
        self.heads.nb       = 1
        self.heads.arch     = [[64]]
        self.heads.actv     = ["selu"]
        self.heads.final    = ["linear"]
        self.k_init         = "lecun_normal"
        self.k_init_final   = "lecun_normal"
        self.b_init         = "zeros"

        # Check inputs
        if hasattr(pms,       "trunk"):        self.trunk        = pms.trunk
        if hasattr(pms.trunk, "arch"):         self.trunk.arch   = pms.trunk.arch
        if hasattr(pms.trunk, "actv"):         self.trunk.actv   = pms.trunk.actv
        if hasattr(pms,       "heads"):        self.heads        = pms.heads
        if hasattr(pms.heads, "nb"):           self.heads.nb     = pms.heads.nb
        if hasattr(pms.heads, "arch"):         self.heads.arch   = pms.heads.arch
        if hasattr(pms.heads, "actv"):         self.heads.actv   = pms.heads.actv
        if hasattr(pms.heads, "final"):        self.heads.final  = pms.heads.final

        # Check that out_dim and heads have same dimension
        if (len(self.out_dim) != pms.heads.nb):
            error("fc", "__init__",
                  "Out_dim and heads should have same dimension")

        # Initialize network
        self.net = []

        # Define trunk
        for l in range(len(self.trunk.arch)):
            self.net.append(Dense(self.trunk.arch[l],
                                  kernel_initializer = self.k_init,
                                  bias_initializer   = self.b_init,
                                  activation         = self.trunk.actv))
            self.net.append(AlphaDropout(0.1))

        # Define heads
        for h in range(self.heads.nb):
            for l in range(len(self.heads.arch[h])):
                self.net.append(Dense(self.heads.arch[h][l],
                                      kernel_initializer = self.k_init,
                                      bias_initializer   = self.b_init,
                                      activation         = self.heads.actv[h]))
                self.net.append(AlphaDropout(0.1))
            self.net.append(Dense(self.out_dim[h],
                                  kernel_initializer = self.k_init_final,
                                  bias_initializer   = self.b_init,
                                  activation         = self.heads.final[h]))

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

        # Compute trunk
        for l in range(2*len(self.trunk.arch)):
            var = self.net[i](var)
            i  += 1

        # Compute heads
        # Each head output is stored in a tf.tensor, and
        # is appended to the global output
        for h in range(self.heads.nb):
            hvar = var
            for l in range(2*len(self.heads.arch[h])):
                hvar = self.net[i](hvar)
                i   += 1
            hvar = self.net[i](hvar)
            i   += 1
            out.append(hvar)

        return out

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
