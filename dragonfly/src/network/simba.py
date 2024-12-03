# Generic imports
from tensorflow.keras.layers    import LayerNormalization

# Custom imports
from dragonfly.src.network.base import *

###############################################
### Simba network class
class simba(base_network):
    def __init__(self, inp_dim, inp_shape, out_dim, pms):

        # Initialize base class
        super().__init__()

        # Set inputs
        self.inp_dim   = inp_dim
        self.inp_shape = inp_shape
        self.out_dim   = out_dim

        # Set default values
        self.n_blocks         = 1
        self.block_size       = 64
        self.expansion_factor = 4
        self.heads            = heads()
        self.heads.nb         = 1
        self.heads.arch       = [[64]]
        self.heads.actv       = ["relu"]
        self.heads.final      = ["linear"]
        self.k_init           = Orthogonal(gain=1.0)
        self.k_init_final     = Orthogonal(gain=0.0)

        # Check inputs
        if hasattr(pms,       "n_blocks"):         self.n_blocks         = pms.n_blocks
        if hasattr(pms,       "block_size"):       self.block_size       = pms.block_size
        if hasattr(pms,       "expansion_factor"): self.expansion_factor = pms.expansion_factor
        if hasattr(pms,       "heads"):            self.heads            = pms.heads
        if hasattr(pms.heads, "nb"):               self.heads.nb         = pms.heads.nb
        if hasattr(pms.heads, "arch"):             self.heads.arch       = pms.heads.arch
        if hasattr(pms.heads, "actv"):             self.heads.actv       = pms.heads.actv
        if hasattr(pms.heads, "final"):            self.heads.final      = pms.heads.final
        if hasattr(pms,       "k_init"):           self.k_init           = pms.k_init
        if hasattr(pms,       "k_init_final"):     self.k_init_final     = pms.k_init_final

        # Check that out_dim and heads have same dimension
        if (len(self.out_dim) != pms.heads.nb):
            error("fc", "__init__",
                  "Out_dim and heads should have same dimension")

        # Initialize network
        self.net = []

        # First linear layer
        self.net.append(Dense(self.block_size,
                              kernel_initializer = self.k_init,
                              activation         = "linear"))

        # Simba block
        for k in range(self.n_blocks):
            self.net.append(LayerNormalization())
            self.net.append(Dense(self.expansion_factor*self.block_size,
                                  kernel_initializer = self.k_init,
                                  activation         = "relu"))
            self.net.append(Dense(self.block_size,
                                  kernel_initializer = self.k_init,
                                  activation         = "linear"))

        # Post-layer normalization
        self.net.append(LayerNormalization())

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
        dummy = self.call(tf.ones([1,self.inp_dim]))

        # Save initial weights
        self.init_weights = self.get_weights()

    # Network forward pass
    @tf.function
    def call(self, var):

        # Initialize
        i   = 0
        out = []

        # First linear lyaer
        var = self.net[i](var)
        i  += 1

        # Simba block
        for k in range(self.n_blocks):
            cvar = self.net[i  ](var)  # LayerNorm
            cvar = self.net[i+1](cvar) # MLP
            cvar = self.net[i+2](cvar) # Linear
            var  = tf.add(cvar, var)
            i   += 3

        # Post-layer normalization
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
            out.append(hvar)

        return out

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
