# Custom imports
from dragonfly.src.policy.tfd import *
from dragonfly.src.policy.base import base_normal
import torch
import torch.nn as nn
import torch.distributions as dist
import math

###############################################
### Normal policy class with full covariance matrix (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal_full(base_normal):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.dim         = self.act_dim
        self.cov_dim     = math.floor(self.dim*(self.dim - 1)/2)
        self.out_dim     = [self.dim, self.dim, self.cov_dim]
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

        if pms.network.heads.final[2] != "sigmoid":
            warning("normal", "__init__",
                    "Final activation for correlations of policy is not sigmoid")

        # Init from base class
        super().__init__(pms)

    # Control (deterministic actions)
    def control(self, obs):
        mu, _, _ = self.forward(obs.float())
        act = mu.detach().cpu().numpy().reshape(-1, self.store_dim)
        return act

    # Compute pdf
    def compute_pdf(self, obs):
        # Get pdf
        mu, sg, cr = self.forward(obs)
        cov = self.get_cov(sg[0], cr[0])

        scl = torch.linalg.cholesky(cov)
        pdf = dist.MultivariateNormal(loc=mu, scale_tril=scl)

        return pdf

    # Compute covariance matrix
    def get_cov(self, sg, cr):
        # Extract sigmas and thetas
        sigmas = sg
        thetas = cr * math.pi

        # Build initial theta matrix
        t = torch.ones(self.dim, self.dim) * math.pi/2.0
        t.diagonal().fill_(0)
        idx = 0
        for dg in range(self.dim-1):
            diag = thetas[idx:idx+self.dim-(dg+1)]
            idx += self.dim-(dg+1)
            t.diagonal(-dg-1).copy_(diag)
        cor = torch.cos(t)

        # Correct upper part to exact zero
        for dg in range(self.dim-1):
            size = self.dim-(dg+1)
            cor.diagonal(dg+1).zero_()

        # Roll and compute additional terms
        for roll in range(self.dim-1):
            vec = torch.ones(self.dim, 1) * math.pi/2
            t = torch.cat([vec, t[:, :self.dim-1]], dim=1)
            for dg in range(self.dim-1):
                t.diagonal(dg+1).zero_()
            cor = cor * torch.sin(t)

        cor = cor @ cor.t()
        scl = torch.diag(torch.sqrt(sigmas))
        cov = scl @ cor @ scl

        return cov

    # Networks forward pass
    def forward(self, state):
        out = self.net(state)
        mu  = out[0]
        sg  = out[1] * self.sigma
        cr  = out[2]

        return mu, sg, cr
