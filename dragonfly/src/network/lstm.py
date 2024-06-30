import torch
import torch.nn as nn
from torch.nn.init import orthogonal_

from dragonfly.src.network.base import *

class LSTM(BaseNetwork):
    def __init__(self, inp_dim, out_dim, pms):
        super(LSTM, self).__init__()

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
        self.seq_length = 1

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
        if hasattr(pms,       "seq_length"):   self.seq_length    = pms.seq_length


        # Check that out_dim and heads have same dimension
        if len(self.out_dim) != pms.heads.nb:
            raise ValueError("Out_dim and heads should have same dimension")

        # Actual observation dimension is inp_dim divided by the sequence length
        self.obs_dim = self.inp_dim // self.seq_length

        # Initialize network
        self.net = nn.ModuleList()

        # Define trunk
        lgt = len(self.trunk.arch)
        for l in range(lgt):
            lstm_layer = nn.LSTM(
                input_size=self.seq_length if l == 0 else self.trunk.arch[l-1],
                hidden_size=self.trunk.arch[l],
                batch_first=True
            )
            self.k_init(lstm_layer.weight_ih_l0)
            self.k_init(lstm_layer.weight_hh_l0)
            self.net.append(lstm_layer)

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
        # Reshape input
        x = x.view(-1, self.obs_dim, self.seq_length)
        out = []
        # Compute trunk
        for l in range(len(self.trunk.arch)):
            x = self.net[l](x)[0]
        
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

    def reset(self):
        for p, w in zip(self.parameters(), self.init_weights):
            p.data.copy_(w)
