# Generic imports
import numpy as np
import torch 

# Custom imports
from dragonfly.src.core.constants import *

###############################################
### Class for generalized advantage estimate
### pms : parameters
class gae():
    def __init__(self, pms):

        # Set default values
        self.gamma      = 0.99
        self.gae_lambda = 0.98
        self.ret_norm   = True

        # Check inputs
        if hasattr(pms, "gamma"):      self.gamma      = pms.gamma
        if hasattr(pms, "gae_lambda"): self.gae_lambda = pms.gae_lambda
        if hasattr(pms, "ret_norm"):   self.ret_norm   = pms.ret_norm

    # Compute GAE
    # rwd : reward array np array
    # val : value array torch tensor
    # nxt : value array shifted by one timestep torch tensor
    # trm : array of terminal values np array
    def compute(self, rwd, val, nxt, trm):

        rwd_ = torch.as_tensor(rwd, dtype=torch.float32)
        trm_ = torch.as_tensor(trm, dtype=torch.float32)

        # Shortcuts
        gm  = self.gamma
        lbd = self.gae_lambda

        # Initialize return and check bootstrapping
        ret = torch.where(trm_ == 2.0, rwd_ + gm * nxt, rwd_)

        # Remove bootstrap information from trm buffer
        tmn = torch.where(trm_ == 2.0, torch.tensor(0.0), trm_)

        # Compute TD residual
        dlt = ret + gm * tmn * nxt - val

        # Compute advantages
        adv = dlt.clone()
        for t in reversed(range(len(adv) - 1)):
            adv[t] += gm * lbd * adv[t + 1]

        # Compute targets
        tgt = adv + val

        # Normalize
        if self.ret_norm:
            adv = (adv - adv.mean()) / (adv.std() + ret_eps)

        return tgt, adv
