import torch
import torch.nn as nn
from torch.nn.init import orthogonal_

from dragonfly.src.network.base import *

class fc(BaseNetwork):
    def __init__(self, inp_dim, out_dim, pms):
        super(fc, self).__init__()
        
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        # Set default values
        self.trunk = trunk()
        self.trunk.arch = [64]
        self.trunk.actv = "relu"
        self.heads = heads()
        self.heads.nb = 1
        self.heads.arch = [[64]]
        self.heads.actv = ["relu"]
        self.heads.final = ["linear"]
        self.k_init = lambda x: orthogonal_(x, gain=1.0)
        self.k_init_final = lambda x: orthogonal_(x, gain=0.0)
        
        # Check inputs
        if hasattr(pms,       "trunk"):        self.trunk        = pms.trunk
        if hasattr(pms.trunk, "arch"):         self.trunk.arch   = pms.trunk.arch
        if hasattr(pms.trunk, "actv"):         self.trunk.actv   = pms.trunk.actv
        if hasattr(pms,       "heads"):        self.heads        = pms.heads
        if hasattr(pms.heads, "nb"):           self.heads.nb     = pms.heads.nb
        if hasattr(pms.heads, "arch"):         self.heads.arch   = pms.heads.arch
        if hasattr(pms.heads, "actv"):         self.heads.actv   = pms.heads.actv
        if hasattr(pms.heads, "final"):        self.heads.final  = pms.heads.final
        if hasattr(pms,       "k_init"):       self.k_init       = pms.k_init
        if hasattr(pms,       "k_init_final"): self.k_init_final = pms.k_init_final

        # Check that out_dim and heads have same dimension
        if len(self.out_dim) != pms.heads.nb:
            raise ValueError("Out_dim and heads should have same dimension")

        # Initialize network
        self.net = nn.ModuleList()

        # Define trunk
        in_features = inp_dim
        for l in range(len(self.trunk.arch)):
            self.net.append(self.create_dense_layer(in_features, self.trunk.arch[l], self.k_init, self.trunk.actv))
            in_features = self.trunk.arch[l]

        # Define heads
        for h in range(self.heads.nb):
            in_features = self.trunk.arch[l]
            for l_ in range(len(self.heads.arch[h])):
                self.net.append(self.create_dense_layer(in_features, self.heads.arch[h][l_], self.k_init, self.heads.actv[h]))
                in_features = self.heads.arch[h][l_]
            self.net.append(self.create_dense_layer(in_features, self.out_dim[h], self.k_init_final, self.heads.final[h]))

        # Initialize weights
        self.init_weights = [p.data.clone() for p in self.parameters()]


    def forward(self, x):
        out = []
        
        # Compute trunk
        for l in range(len(self.trunk.arch)):
            x = self.net[l](x)
        
        # Compute heads
        i = len(self.trunk.arch)
        for h in range(self.heads.nb):
            hx = x
            for l in range(len(self.heads.arch[h])):
                hx = self.net[i](hx)
                i += 1
            hx = self.net[i](hx)
            i += 1
            out.append(hx)
        
        return out

    def reset(self):
        for p, w in zip(self.parameters(), self.init_weights):
            p.data.copy_(w)
