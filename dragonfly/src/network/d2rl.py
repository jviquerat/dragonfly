from dragonfly.src.network.base import *


class d2rl(base_network):
    def __init__(self, inp_dim, out_dim, pms):

        # Initialize base class
        super().__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.trunk = pms.trunk if hasattr(pms, "trunk") else trunk()
        self.trunk.arch = pms.trunk.arch if hasattr(pms.trunk, "arch") else [64]
        self.trunk.actv = pms.trunk.actv if hasattr(pms.trunk, "actv") else "relu"
        self.heads = pms.heads if hasattr(pms.heads, "heads") else heads()
        self.heads.nb = pms.heads.nb if hasattr(pms.heads, "nb") else 1
        self.heads.arch = pms.heads.arch if hasattr(pms.heads, "arch") else [[64]]
        self.heads.actv = pms.heads.actv if hasattr(pms.heads, "actv") else ["relu"]
        self.heads.final = (
            pms.heads.final if hasattr(pms.heads, "final") else ["linear"]
        )
        self.k_init = pms.k_init if hasattr(pms, "k_init") else Orthogonal(gain=1.0)
        self.k_init_final = (
            pms.k_init_final if hasattr(pms, "k_init_final") else Orthogonal(gain=0.0)
        )

        # Check that out_dim and heads have same dimension
        if len(self.out_dim) != pms.heads.nb:
            error("d2rl", "__init__", "Out_dim and heads should have same dimension")

        # Initialize network
        self.net = []

        # Define trunk
        for l in range(len(self.trunk.arch)):
            self.net.append(
                Dense(
                    self.trunk.arch[l],
                    kernel_initializer=self.k_init,
                    activation=self.trunk.actv,
                )
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
    def call(self, input):
        # Initialize
        i = 0
        out = []
        var = tf.identity(input)

        # Compute trunk
        for _ in range(len(self.trunk.arch)):
            var = self.net[i](var)
            # Concatenation for D2RL
            var = tf.concat([var, input], 0)
            i += 1

        # Compute heads
        # Each head output is stored in a tf.tensor, and
        # is appended to the global output
        for h in range(self.heads.nb):
            hvar = var
            for _ in range(len(self.heads.arch[h])):
                hvar = self.net[i](hvar)
                # Concatenation for D2RL
                hvar = tf.concat([hvar, input], 0)
                i += 1
            hvar = self.net[i](hvar)
            i += 1
            out.append(hvar)

        return out

    # Reset weights
    def reset(self):
        self.set_weights(self.init_weights)
