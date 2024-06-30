# Custom imports
from dragonfly.src.core.constants import *
from dragonfly.src.policy.tfd import *
from dragonfly.src.policy.base import base_normal
import torch
import torch.nn as nn
import math

###############################################
### Tanh-normal policy class (continuous)
class tanh_normal(base_normal):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.dim         = self.act_dim
        self.out_dim     = [self.dim, self.dim]
        self.store_dim   = self.act_dim
        self.store_type  = float
        self.target      = target
        self.min_log_std = pms.min_log_std
        self.max_log_std = pms.max_log_std

        # Check parameters
        if pms.network.heads.final[0] != "linear":
            warning("tanh_normal", "__init__",
                    "Final activation for mean of policy is not linear")

        if pms.network.heads.final[1] != "linear":
            warning("tanh_normal", "__init__",
                    "Final activation for stddev of policy is not linear")

        # Init from base class
        super().__init__(pms)

    # Control (deterministic actions)
    def control(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.from_numpy(obs).to(torch.float32)
        else:
            obs_tensor = obs
        mu, _   = self.forward(obs_tensor)
        act     = mu.reshape(-1, self.store_dim)
        tanh_act = torch.tanh(act)
        tanh_act = tanh_act.detach().cpu().numpy().reshape(-1, self.store_dim)
        return tanh_act

    # Networks forward pass
    def forward(self, state):
        out     = self.net(state)
        mu      = out[0]
        log_std = out[1]
        log_std = torch.clamp(log_std,
                              self.min_log_std,
                              self.max_log_std)
        std     = torch.exp(log_std)
        return mu, std

    # Sample actions
    # Mostly taken from openAI implementation
    def sample(self, obs):
        # Reparameterization trick
        mu, std = self.forward(obs)
        act     = mu + torch.randn_like(mu) * std

        # Compute gaussian likelihood
        lkh = -0.5 * (((act - mu) / (std + eps)) ** 2 +
                      2.0 * torch.log(std) +
                      math.log(2.0 * math.pi))
        lgp = torch.sum(lkh, dim=1)

        # Squash actions
        tanh_act = torch.tanh(act)

        # Compute log-prob of reparameterized action
        # OpenAI version, numerically stable
        sth  = 2.0 * (math.log(2.0) - act - torch.nn.functional.softplus(-2.0 * act))
        sth  = torch.sum(sth, dim=1)
        lgp -= sth

        return tanh_act, lgp
