# Custom imports
from dragonfly.src.network.base import *
from dragonfly.src.utils.rmsnorm import RMSNorm


###############################################
### Gated Fully-connected network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class gated_fc(base_network):
    def __init__(self, inp_dim, out_dim, pms):
        """
        We consider that the block of the network is not a simple dense layer,
        but a gated block made of 3 dense layers with a RMSNorm layer beforehand.
        We also remplace the `relu` activation function by a `gelu` and add a residual
        addition between each inputs processing (like https://arxiv.org/pdf/2402.19427.pdf)
        """

        # Initialize base class
        super().__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Set default values
        self.trunk = trunk()
        self.trunk.arch = [64]
        self.trunk.actv = "gelu"
        self.heads = heads()
        self.heads.nb = 1
        self.heads.arch = [[64]]
        self.heads.actv = ["relu"]
        self.heads.final = ["linear"]
        self.k_init = Orthogonal(gain=1.0)
        self.k_init_final = Orthogonal(gain=0.0)
        self.expansion_factor = 3

        # Check inputs
        if hasattr(pms, "trunk"):
            self.trunk = pms.trunk
        if hasattr(pms.trunk, "arch"):
            self.trunk.arch = pms.trunk.arch
        if hasattr(pms.trunk, "actv"):
            self.trunk.actv = pms.trunk.actv
        if hasattr(pms, "heads"):
            self.heads = pms.heads
        if hasattr(pms.heads, "nb"):
            self.heads.nb = pms.heads.nb
        if hasattr(pms.heads, "arch"):
            self.heads.arch = pms.heads.arch
        if hasattr(pms.heads, "actv"):
            self.heads.actv = pms.heads.actv
        if hasattr(pms.heads, "final"):
            self.heads.final = pms.heads.final
        if hasattr(pms, "k_init"):
            self.k_init = pms.k_init
        if hasattr(pms, "k_init_final"):
            self.k_init_final = pms.k_init_final
        if hasattr(pms, "expansion_factor"):
            self.expansion_factor = pms.expansion_factor

        # Check that out_dim and heads have same dimension
        if len(self.out_dim) != pms.heads.nb:
            error("fc", "__init__", "Out_dim and heads should have same dimension")

        # Initialize network
        self.net = []

        def _build_dense_layer(layer_index, expansion_factor=1, use_activation=True):
            activation = self.trunk.actv if use_activation else None
            return Dense(
                self.trunk.arch[layer_index] * expansion_factor,
                kernel_initializer=self.k_init,
                activation=activation,
            )

        # Define trunk
        for l in range(len(self.trunk.arch)):
            self.net.append(
                {
                    "left": _build_dense_layer(
                        l, expansion_factor=self.expansion_factor
                    ),
                    "right": _build_dense_layer(
                        l, expansion_factor=self.expansion_factor, use_activation=False
                    ),
                    "middle": _build_dense_layer(l, use_activation=False),
                    "norm": RMSNorm(),
                }
            )

        # Define heads
        for h in range(self.heads.nb):
            for l in range(len(self.heads.arch[h])):
                self.net.append(
                    Dense(
                        self.heads.arch[h][l],
                        kernel_initializer=self.k_init,
                        activation=self.heads.actv[h],
                    )
                )
            self.net.append(
                Dense(
                    self.out_dim[h],
                    kernel_initializer=self.k_init_final,
                    activation=self.heads.final[h],
                )
            )

        # Initialize weights
        dummy = self.call(tf.ones([1, self.inp_dim]))

        # Save initial weights
        self.init_weights = self.get_weights()

    # Network forward pass
    @tf.function
    def call(self, var):

        # Initialize
        i = 0
        out = []

        def _gated_layer(x, layer: dict):
            normalized_x = layer["norm"](x)
            left_x = layer["left"](normalized_x)
            right_x = layer["right"](normalized_x)
            return layer["middle"](left_x * right_x)

        # Compute trunk
        for l in range(len(self.trunk.arch)):
            var = var + _gated_layer(var, self.net[i])
            i += 1

        # Compute heads
        # Each head output is stored in a tf.tensor, and
        # is appended to the global output
        for h in range(self.heads.nb):
            hvar = var
            for l in range(len(self.heads.arch[h])):
                hvar = self.net[i](hvar)
                i += 1
            hvar = self.net[i](hvar)
            i += 1
            out.append(hvar)

        return out

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
