import torch
import torch.nn as nn
from torch.nn.init import orthogonal_

from dragonfly.src.network.base import *


class fc(BaseNetwork):
    def __init__(self, inp_dim, out_dim, pms, agent_type = AgentType.ON_POLICY):
        super(fc, self).__init__(inp_dim, out_dim, agent_type)

        # Check inputs
        self._fetch_input(pms=pms)

        # Check that out_dim and heads have same dimension
        if len(self.out_dim) != pms.heads.nb:
            raise ValueError("Out_dim and heads should have same dimension")

        # Initialize network
        self.trunk_net = self._build_trunk()
        self._build_heads()
        # Initialize weights
        self.init_weights = [p.data.clone() for p in self.parameters()]

    def forward(self, x):
        x = self.trunk_net(x)
        out = [getattr(self, f"head_{h}")(x) for h in range(self.heads.nb)]
        return out

    def reset(self):
        for p, w in zip(self.parameters(), self.init_weights):
            p.data.copy_(w)
