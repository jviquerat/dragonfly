# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.core.constants import *

###############################################
### Class for advantage
### pms : parameters
class advantage():
    def __init__(self, pms):

        # Set default values
        self.gamma    = 0.99
        self.ret_norm = True

        # Check inputs
        if hasattr(pms, "gamma"):    self.gamma    = pms.gamma
        if hasattr(pms, "ret_norm"): self.ret_norm = pms.ret_norm

    # Compute advantage
    # rwd : reward array
    # val : value array
    # nxt : value array shifted by one timestep
    # trm : array of terminal values
    def compute(self, rwd, val, nxt, trm):

        # Shortcuts
        gm = self.gamma

        # Initialize return and check bootstrapping
        ret = np.where(trm == 2.0, rwd + gm*nxt, rwd)

        # Remove bootstrap information from trm buffer
        trm = np.where(trm == 2.0, 0.0, trm)

        # Return as discounted sum
        for t in reversed(range(len(ret)-1)):
            ret[t] += trm[t]*gm*ret[t+1]

        # Advantage
        adv = ret - val

        # Compute targets
        tgt = ret.copy()

        # Normalize
        if self.ret_norm:
            adv = (adv-np.mean(adv))/(np.std(adv) + ret_eps)

        return tgt, adv
