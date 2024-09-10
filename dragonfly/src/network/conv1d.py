# Generic imports
import torch
import torch.nn as nn

# Custom imports
from dragonfly.src.network.base import *


###############################################
### Fully-connected network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class conv1d(BaseNetwork):
    def __init__(self, inp_dim, out_dim, pms):

        # Initialize base class
        super(conv1d, self).__init__(inp_dim, out_dim)

        # Set default values
        self.trunk.k_size = [4]
        self.trunk.strides = [2]
        self.original_dim = [inp_dim, 1]

        # Check inputs
        self._fetch_input(pms=pms, conv=True)

        # Check that out_dim and heads have same dimension
        if len(self.out_dim) != pms.heads.nb:
            error("conv1d", "__init__", "Out_dim and heads should have same dimension")

        # Define trunk
        new_hidden_sizes = [1] + self.trunk.arch
        layers = [
            nn.Sequential(
                nn.Conv1d(
                    in_channels=new_hidden_sizes[k],
                    out_channels=new_hidden_sizes[k + 1],
                    kernel_size=self.trunk.k_size[k],
                    stride=self.trunk.stride[k],
                    padding=0,
                ),
                get_activation(self.trunk.actv),
            )
            for k in range(0, len(new_hidden_sizes) - 1)
        ]
        self.trunk_net = nn.Sequential(*layers)

        # Define heads
        self._build_heads()

        # Initialize weights
        self.apply(self._init_weights)

        # Save initial weights
        self.init_weights = self.state_dict()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            self.k_init(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear) and module.out_features in self.out_dim:
            self.k_init_final(module.weight)

    def forward(self, x):
        x = x.view(-1, 1, self.original_dim[0])

        x = self.trunk_net(x)
        x = x.flatten(1)
        out = [
            getattr(self, f"head_{h}")(x).view(-1, self.out_dim[h])
            for h in range(self.heads.nb)
        ]
        return out

    # Reset weights
    def reset(self):
        self.load_state_dict(self.init_weights)
