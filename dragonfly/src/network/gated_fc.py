# Custom imports
import torch
import torch.nn as nn
from torch.nn.init import orthogonal_
from dragonfly.src.utils.rmsnorm import RMSNorm
from dragonfly.src.network.base import *

class GatedFC(BaseNetwork):
    def __init__(self, inp_dim, out_dim, pms):
        super(GatedFC, self).__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Set default values
        self.trunk = trunk()
        self.trunk.arch = [64]
        self.trunk.actv = "relu"
        self.heads = heads()
        self.heads_nb = 1
        self.heads_arch = [[64]]
        self.heads_actv = ["relu"]
        self.heads_final = ["linear"]
        self.k_init = lambda x: orthogonal_(x, gain=1.0)
        self.k_init_final = lambda x: orthogonal_(x, gain=0.0)
        self.expansion_factor = 3

        # Check inputs and update values if provided in pms
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

        def _build_dense_layer(in_features, out_features, activation=None):
            layer = nn.Linear(in_features, out_features)
            self.k_init(layer.weight)
            return nn.Sequential(layer, activation) if activation else layer
        
        # Define trunk
        for l in range(len(self.trunk.arch)):
            in_features = inp_dim if l == 0 else self.trunk.arch[l-1]
            self.net.append(nn.ModuleDict({
                'left': _build_dense_layer(in_features, self.trunk.arch[l] * self.expansion_factor, get_activation(self.trunk.actv)),
                'right': _build_dense_layer(in_features, self.trunk.arch[l] * self.expansion_factor),
                'middle': _build_dense_layer(self.trunk.arch[l] * self.expansion_factor, self.trunk.arch[l]),
                'norm': RMSNorm(in_features),
            }))

        # Define heads
        for h in range(self.heads.nb):
            in_features = self.trunk.arch[l]
            for l_ in range(len(self.heads.arch[h])):
                self.net.append(self.create_dense_layer(in_features, self.heads.arch[h][l_], self.k_init, self.heads.actv[h]))
                in_features = self.heads.arch[h][l_]
            self.net.append(self.create_dense_layer(in_features, self.out_dim[h], self.k_init_final, self.heads.final[h]))

        # Save initial weights
        self.init_weights = [p.data.clone() for p in self.parameters()]

    def forward(self, x):
        # Compute trunk
        print(self.net)
        for l, layer in enumerate(self.net):
            if isinstance(layer, nn.ModuleDict):
                normalized_x = layer['norm'](x)
                left_x = layer['left'](normalized_x)
                right_x = layer['right'](normalized_x)
                gated_output = layer['middle'](left_x * right_x)
                x = x + gated_output if l > 0 else gated_output
            else:
                break

        # Compute heads
        out = []
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
