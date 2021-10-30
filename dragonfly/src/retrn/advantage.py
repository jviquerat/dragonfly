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
        self.ret_clip = True

        # Check inputs
        if hasattr(pms, "gamma"):    self.gamma    = pms.gamma
        if hasattr(pms, "ret_norm"): self.ret_norm = pms.ret_norm
        if hasattr(pms, "ret_clip"): self.ret_clip = pms.ret_clip

    # Compute advantage
    # rwd : reward array
    # val : value array
    # nxt : value array shifted by one timestep
    # trm : array of terminal values
    # bts : array of bootstraping tags
    def compute(self, rwd, val, nxt, trm, bts):

        # Shortcuts
        gm = self.gamma

        # Initialize return and check bootstrapping
        ret = np.where(bts == 1.0, rwd + gm*nxt, rwd)

        # Return as discounted sum
        for t in reversed(range(len(ret)-1)):
            ret[t] += trm[t]*gm*ret[t+1]

        # Advantage
        adv = ret - val

        # Compute targets
        tgt = ret.copy()

        # Normalize
        if self.ret_norm: adv = (adv-np.mean(adv))/(np.std(adv) + ret_eps)

        # Clip if required
        if self.ret_clip: adv = np.maximum(adv, 0.0)

        return tgt, adv
