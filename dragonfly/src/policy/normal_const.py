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
class normal_const(base_normal):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.dim         = self.act_dim
        self.out_dim     = [self.dim]
        self.store_dim   = self.act_dim
        self.store_type  = float
        self.sigma       = pms.sigma
        self.target      = target

        # Check parameters
        if pms.network.heads.final[0] != "tanh":
            warning("normal", "__init__",
                    "Final activation for mean of policy is not tanh")

        # Init from base class
        super().__init__(pms)

    # Control (deterministic actions)
    def control(self, obs):
        mu  = self.forward(obs.float())
        act = mu.detach().cpu().numpy().reshape(-1, self.store_dim)

        return act

    # Compute pdf
    def compute_pdf(self, obs):
        # Get pdf
        mu    = self.forward(obs)
        sg    = torch.tensor([[self.sigma]])
        sigma = sg.repeat(1, self.dim)
        pdf   = dist.MultivariateNormal(loc=mu,
                                        scale_tril=torch.diag(sigma.squeeze()))

        return pdf

    # Networks forward pass
    def forward(self, state):
        return self.net(state)[0]
