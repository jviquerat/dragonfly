# Custom imports
from dragonfly.src.policy.tfd import *
from dragonfly.src.policy.base import base_normal
import torch
import torch.nn as nn
import torch.distributions as dist

###############################################
### Normal policy class with isotropic covariance matrix (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal_iso(base_normal):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.dim         = self.act_dim
        self.out_dim     = [self.dim, 1]
        self.store_dim   = self.act_dim
        self.store_type  = float
        self.target      = target

        self.sigma       = 1.0
        if hasattr(pms, "sigma"): self.sigma = pms.sigma

        # Check parameters
        if pms.network.heads.final[0] != "tanh":
            warning("normal", "__init__",
                    "Final activation for mean of policy is not tanh")

        if pms.network.heads.final[1] != "sigmoid":
            warning("normal", "__init__",
                    "Final activation for stddev of policy is not sigmoid")

        # Init from base class
        super().__init__(pms)

    # Control (deterministic actions)
    def control(self, obs):
        mu, _ = self.forward(obs)
        act = mu.detach().cpu().numpy().reshape(-1, self.store_dim)
        return act

    # Compute pdf
    def compute_pdf(self, obs):
        # Get pdf
        mu, sg = self.forward(obs)
        sigma = sg.repeat(1, self.dim)
        pdf = dist.MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(sigma))
        return pdf

    # Networks forward pass
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.from_numpy(state).to(torch.float32)
        else:
            state_tensor = state
        out = self.net(state_tensor)
        mu  = out[0]
        sg  = out[1] * self.sigma
        return mu, sg
