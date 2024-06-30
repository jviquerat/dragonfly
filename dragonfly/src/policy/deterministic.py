# Custom imports
from dragonfly.src.policy.tfd import *
from dragonfly.src.policy.base import base_policy
import torch
import torch.nn as nn

###############################################
### Deterministic policy class (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class deterministic(base_policy):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.out_dim    = [self.dim]
        self.store_dim  = self.act_dim
        self.store_type = float
        self.target     = target

        # Define and init network
        if pms.network.heads.final[0] != "tanh":
            warning("normal", "__init__",
                    "Final activation for network of deterministic policy is not tanh")

        # Init from base class
        super().__init__(pms)

    # Get actions
    def actions(self, obs):
        if isinstance(obs, torch.Tensor):
            act = self.forward(obs)
        else:
            act = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        act = act.detach().cpu().numpy().reshape(-1, self.store_dim)

        return act

    # Control (deterministic actions)
    def control(self, obs):
        return self.actions(obs)

    # Networks forward pass
    def forward(self, state):
        out = self.net(state)[0]
        return out
