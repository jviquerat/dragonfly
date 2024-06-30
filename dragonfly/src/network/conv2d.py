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
        super(conv2d, self).__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Set default values
        self.trunk         = trunk()
        self.trunk.filters = [64]
        self.trunk.kernels = [3]
        self.trunk.stride  = 1
        self.trunk.actv    = nn.ReLU()
        self.heads         = heads()
        self.heads.nb      = 1
        self.heads.arch    = [[64]]
        self.heads.actv    = [nn.ReLU()]
        self.heads.final   = [nn.Identity()]
        self.k_init        = nn.init.orthogonal_
        self.k_init_final  = lambda x: nn.init.orthogonal_(x, gain=0.0)
        self.original_dim  = None
        self.pooling       = False

        # Check inputs
        if hasattr(pms,       "trunk"):        self.trunk         = pms.trunk
        if hasattr(pms.trunk, "filters"):      self.trunk.filters = pms.trunk.filters
        if hasattr(pms.trunk, "kernels"):      self.trunk.kernels = pms.trunk.kernels
        if hasattr(pms.trunk, "strides"):      self.trunk.stride  = pms.trunk.stride
        if hasattr(pms.trunk, "actv"):         self.trunk.actv    = pms.trunk.actv
        if hasattr(pms,       "heads"):        self.heads         = pms.heads
        if hasattr(pms.heads, "nb"):           self.heads.nb      = pms.heads.nb
        if hasattr(pms.heads, "arch"):         self.heads.arch    = pms.heads.arch
        if hasattr(pms.heads, "actv"):         self.heads.actv    = pms.heads.actv
        if hasattr(pms.heads, "final"):        self.heads.final   = pms.heads.final
        if hasattr(pms,       "k_init"):       self.k_init        = pms.k_init
        if hasattr(pms,       "k_init_final"): self.k_init_final  = pms.k_init_final
        if hasattr(pms,       "original_dim"): self.original_dim  = pms.original_dim
        if hasattr(pms,       "pooling"):      self.pooling       = pms.pooling

        # Specific dimensions
        self.nx    = self.original_dim[0]
        self.ny    = self.original_dim[1]
        self.stack = self.original_dim[2]

        # Check that out_dim and heads have same dimension
        if (len(self.out_dim) != pms.heads.nb):
            error("conv2d", "__init__",
                  "Out_dim and heads should have same dimension")

        # Initialize network
        self.net = nn.ModuleList()

        assert len(self.trunk.filters) == len(
            self.trunk.kernels
        ), f"Wrong kernel list. Expected {len(self.trunk.filters)} elements, got {len(self.trunk.kernels)}."

        # Define trunk
        for l in range(len(self.trunk.filters)):
            if l == 0:
                layer_conv = nn.Conv2d(in_channels=self.stack,
                                          out_channels=self.trunk.filters[l],
                                          kernel_size=self.trunk.kernels[l],
                                          stride=self.trunk.stride,
                                          padding='same')
            else:
                layer_conv = nn.Conv2d(in_channels=self.trunk.filters[l-1],
                                          out_channels=self.trunk.filters[l],
                                          kernel_size=self.trunk.kernels[l],
                                          stride=self.trunk.stride,
                                          padding='same')

            self.net.append(
                nn.Sequential(
                    layer_conv,
                    get_activation(self.trunk.actv)
                )
            )

            if self.pooling:
                self.net.append(nn.MaxPool2d(kernel_size=2))

        # Define heads
        for h in range(self.heads.nb):
            in_features = self.trunk.arch[l]
            for l_ in range(len(self.heads.arch[h])):
                self.net.append(self.create_dense_layer(in_features, self.heads.arch[h][l_], self.k_init, self.heads.actv[h]))
                in_features = self.heads.arch[h][l_]
            self.net.append(self.create_dense_layer(in_features, self.out_dim[h], self.k_init_final, self.heads.final[h]))

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
        # Reshape to original dimension
        var = var.view(-1, self.stack, self.nx, self.ny)

        # Compute trunk
        for layer in self.net[:3*len(self.trunk.filters) if self.pooling else 2*len(self.trunk.filters)]:
            var = layer(var)

        var = var.flatten(1)

        # Compute heads
        out = []
        start = 3*len(self.trunk.filters) if self.pooling else 2*len(self.trunk.filters)
        for h in range(self.heads.nb):
            hvar = var
            for layer in self.net[start:start+2*len(self.heads.arch[h])+2]:
                hvar = layer(hvar)
            out.append(hvar)
            start += 2*len(self.heads.arch[h])+2

        return out

    # Reset weights
    def reset(self):
        self.load_state_dict(self.init_weights)
