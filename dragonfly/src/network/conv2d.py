# Generic imports
import torch
import torch.nn as nn

# Custom imports
from dragonfly.src.network.base import *


###############################################
### CNN class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### pms      : network parameters
class conv2d(BaseNetwork):
    def __init__(self, inp_dim, out_dim, pms):

        # Initialize base class
        super(conv2d, self).__init__(inp_dim, out_dim)

        # Set default values
        self.trunk.filters = [64]
        self.trunk.kernels = [3]
        self.trunk.stride = 1
        self.original_dim = None
        self.pooling = False

        # Check inputs
        self._fetch_input(pms=pms, conv=True)

        # Specific dimensions
        self.nx = self.original_dim[0]
        self.ny = self.original_dim[1]
        self.stack = self.original_dim[2]

        # Check that out_dim and heads have same dimension
        if len(self.out_dim) != pms.heads.nb:
            error("conv2d", "__init__", "Out_dim and heads should have same dimension")

        assert len(self.trunk.filters) == len(
            self.trunk.kernels
        ), f"Wrong kernel list. Expected {len(self.trunk.filters)} elements, got {len(self.trunk.kernels)}."

        # Define trunk
        new_hidden_sizes = [self.stack] + self.trunk.filters
        layers = []
        for k in range(0, len(new_hidden_sizes) - 1):
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=new_hidden_sizes[k],
                    out_channels=new_hidden_sizes[k + 1],
                    kernel_size=self.trunk.kernels[k],
                    stride=self.trunk.stride,
                    padding="same",
                ),
                get_activation(self.trunk.actv),
            )
            if self.pooling:
                layer.append(nn.MaxPool2d(kernel_size=2))
            layers.append(layer)

        self.trunk_net = nn.Sequential(*layers)
        # Define head
        self.trunk.arch = [9216]
        self._build_heads()

        # Initialize weights
        self.apply(self._init_weights)

        # Save initial weights
        self.init_weights = self.state_dict()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            self.k_init(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear) and module.out_features in self.out_dim:
            self.k_init_final(module.weight)

    # Network forward pass
    def forward(self, var):
        var = var.view(-1, self.stack, self.nx, self.ny)
        var = self.trunk_net(var)
        var = var.flatten(1)
        out = [getattr(self, f"head_{h}")(var) for h in range(self.heads.nb)]
        return out

    # Reset weights
    def reset(self):
        self.load_state_dict(self.init_weights)
