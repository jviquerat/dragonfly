# Generic imports
from tensorflow.keras.layers    import LSTM, Permute

# Custom imports
from dragonfly.src.network.base import *

###############################################
### Fully-connected network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class lstm(base_network):
    def __init__(self, inp_dim, out_dim, pms):

        # Initialize base class
        super().__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Set default values
        self.trunk          = trunk()
        self.trunk.arch     = [64]
        self.trunk.actv     = "relu"
        self.heads          = heads()
        self.heads.nb       = 1
        self.heads.arch     = [[64]]
        self.heads.actv     = ["relu"]
        self.heads.final    = ["linear"]
        self.k_init         = Orthogonal(gain=1.0)
        self.k_init_final   = Orthogonal(gain=0.0)
        self.seq_length     = 1

        # Check inputs
        if hasattr(pms,       "trunk"):        self.trunk         = pms.trunk
        if hasattr(pms.trunk, "arch"):         self.trunk.arch    = pms.trunk.arch
        if hasattr(pms.trunk, "actv"):         self.trunk.actv    = pms.trunk.actv
        if hasattr(pms,       "heads"):        self.heads         = pms.heads
        if hasattr(pms.heads, "nb"):           self.heads.nb      = pms.heads.nb
        if hasattr(pms.heads, "arch"):         self.heads.arch    = pms.heads.arch
        if hasattr(pms.heads, "actv"):         self.heads.actv    = pms.heads.actv
        if hasattr(pms.heads, "final"):        self.heads.final   = pms.heads.final
        if hasattr(pms,       "k_init"):       self.k_init        = pms.k_init
        if hasattr(pms,       "k_init_final"): self.k_init_final  = pms.k_init_final
        if hasattr(pms,       "seq_length"):   self.seq_length    = pms.seq_length

        # Actual observation dimension is inp_dim divided by the sequence length
        self.obs_dim = self.inp_dim//self.seq_length

        # Check that out_dim and heads have same dimension
        if (len(self.out_dim) != pms.heads.nb):
            error("conv1d", "__init__",
                  "Out_dim and heads should have same dimension")

        # Initialize network
        self.net = []

        # Define trunk
        lgt = len(self.trunk.arch)
        for l in range(lgt):

            # Handle return_sequence depending on network depth
            # return_sequence must be set to true for stacked lstms,
            # for all but last layer
            return_seq = False
            if (l < lgt-1): return_seq = True

            # Handle input_shape depending on network depth
            if (l == 0):
                self.net.append(LSTM(units              = self.trunk.arch[l],
                                     activation         = self.trunk.actv,
                                     kernel_initializer = self.k_init,
                                     return_sequences   = return_seq,
                                     input_shape        = (self.obs_dim, self.seq_length)))
            else:
                self.net.append(LSTM(units              = self.trunk.arch[l],
                                     activation         = self.trunk.actv,
                                     kernel_initializer = self.k_init,
                                     return_sequences   = return_seq))

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
        dummy = self.call(tf.ones([1, self.obs_dim, self.seq_length]))

        # Save initial weights
        self.init_weights = self.get_weights()

    # Network forward pass
    @tf.function
    def call(self, var):

        # Initialize
        i   = 0
        out = []

        var = Reshape([self.obs_dim, self.seq_length])(var)

        # Compute trunk
        for l in range(len(self.trunk.arch)):
            var = self.net[i](var)
            i  += 1

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
            hvar = Reshape([self.out_dim[h]])(hvar)
            out.append(hvar)

        return out

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
