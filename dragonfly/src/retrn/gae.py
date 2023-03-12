# Generic imports
import numpy as np

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
    # rwd : reward array
    # val : value array
    # nxt : value array shifted by one timestep
    # trm : array of terminal values
    def compute(self, rwd, val, nxt, trm):

        # Shortcuts
        gm  = self.gamma
        lbd = self.gae_lambda

        # Initialize return and check bootstrapping
        ret = np.where(trm == 2.0, rwd + gm*nxt, rwd)

        # Remove bootstrap information from trm buffer
        trm = np.where(trm == 2.0, 0.0, trm)

        # Compute TD residual
        dlt    = np.zeros_like(ret)
        dlt[:] = ret[:] + gm*trm[:]*nxt[:] - val[:]

        # Compute advantages
        adv = np.zeros_like(dlt)
        for t in reversed(range(len(adv)-1)):
            adv[t] = dlt[t] + trm[t]*gm*lbd*adv[t+1]

        # Compute targets
        tgt  = adv.copy()
        tgt += val

        # Normalize
        if self.ret_norm:
            adv = (adv-np.mean(adv))/(np.std(adv) + ret_eps)

        return tgt, adv
