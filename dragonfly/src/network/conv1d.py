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
        super(conv1d, self).__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Set default values
        self.trunk          = trunk()
        self.trunk.arch     = [64]
        self.trunk.k_size   = [4]
        self.trunk.strides  = [2]
        self.trunk.actv     = nn.ReLU()
        self.heads          = heads()
        self.heads.nb       = 1
        self.heads.arch     = [[64]]
        self.heads.actv     = [nn.ReLU()]
        self.heads.final    = [nn.Identity()]
        self.k_init         = nn.init.orthogonal_
        self.k_init_final   = lambda x: nn.init.orthogonal_(x, gain=0.0)
        self.original_dim   = [inp_dim, 1]

        # Check inputs
        if hasattr(pms,       "trunk"):        self.trunk         = pms.trunk
        if hasattr(pms.trunk, "arch"):         self.trunk.arch    = pms.trunk.arch
        if hasattr(pms.trunk, "k_size"):       self.trunk.k_size  = pms.trunk.k_size
        if hasattr(pms.trunk, "strides"):      self.trunk.strides = pms.trunk.strides
        if hasattr(pms.trunk, "actv"):         self.trunk.actv    = pms.trunk.actv
        if hasattr(pms,       "heads"):        self.heads         = pms.heads
        if hasattr(pms.heads, "nb"):           self.heads.nb      = pms.heads.nb
        if hasattr(pms.heads, "arch"):         self.heads.arch    = pms.heads.arch
        if hasattr(pms.heads, "actv"):         self.heads.actv    = pms.heads.actv
        if hasattr(pms.heads, "final"):        self.heads.final   = pms.heads.final
        if hasattr(pms,       "k_init"):       self.k_init        = pms.k_init
        if hasattr(pms,       "k_init_final"): self.k_init_final  = pms.k_init_final
        if hasattr(pms,       "original_dim"): self.original_dim  = pms.original_dim

        # Check that out_dim and heads have same dimension
        if (len(self.out_dim) != pms.heads.nb):
            error("conv1d", "__init__",
                  "Out_dim and heads should have same dimension")

        # Initialize network
        self.net = nn.ModuleList()

        # Define trunk
        for l in range(len(self.trunk.arch)):
            self.net.append(
                nn.Sequential(
                    nn.Conv1d(in_channels = 1 if l == 0 else self.trunk.arch[l-1],
                                      out_channels = self.trunk.arch[l],
                                      kernel_size = self.trunk.k_size[l],
                                      stride = self.trunk.strides[l],
                                      padding = 0),
                    get_activation(self.trunk.actv)
                )
            )

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
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            self.k_init(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear) and module.out_features in self.out_dim:
            self.k_init_final(module.weight)

    def forward(self, x):
        out = []
        x = x.view(-1, 1, self.original_dim[0])
        
        # Compute trunk
        for l in range(len(self.trunk.arch)):
            x = self.net[l](x)
        x = x.flatten(1)
        # Compute heads
        i = len(self.trunk.arch)
        for h in range(self.heads.nb):
            hx = x
            for l in range(len(self.heads.arch[h])):
                hx = self.net[i](hx)
                i += 1
            hx = self.net[i](hx)
            i += 1
            out.append(hx.view(-1, self.out_dim[h]))    

        return out


    # Reset weights
    def reset(self):
        self.load_state_dict(self.init_weights)
