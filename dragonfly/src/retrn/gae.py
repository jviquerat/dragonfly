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
        self.gae_lambda = 0.99
        self.ret_norm   = True
        self.ret_clip   = True

        # Check inputs
        if hasattr(pms, "gamma"):      self.gamma      = pms.gamma
        if hasattr(pms, "gae_lambda"): self.gae_lambda = pms.gae_lambda
        if hasattr(pms, "ret_norm"):   self.ret_norm   = pms.ret_norm
        if hasattr(pms, "ret_clip"):   self.ret_clip   = pms.ret_clip

    # Compute GAE
    # rwd : reward array
    # val : value array
    # nxt : value array shifted by one timestep
    # trm : array of terminal values
    def compute(self, rwd, val, nxt, trm):

        # Shortcuts
        gm  = self.gamma
        lbd = self.gae_lambda

        # Handle mask from termination signals
        msk = np.zeros(len(trm))
        for i in range(len(trm)):
            if (trm[i] == 0): msk[i] = 1.0
            if (trm[i] == 1): msk[i] = 0.0
            if (trm[i] == 2): msk[i] = 1.0

        # Compute deltas
        buff = zip(rwd, msk, nxt, val)
        dlt  = [r + gm*m*nv - v for r, m, nv, v in buff]
        dlt  = np.stack(dlt)

        # Modify termination mask for GAE
        msk2 = np.zeros(len(trm))
        for i in range(len(trm)):
            if (trm[i] == 0): msk2[i] = 1.0
            if (trm[i] == 1): msk2[i] = 0.0
            if (trm[i] == 2): msk2[i] = 0.0

        # Compute advantages
        adv = dlt.copy()
        for t in reversed(range(len(adv)-1)):
            adv[t] += msk2[t]*gm*lbd*adv[t+1]

        # Compute targets
        tgt  = adv.copy()
        tgt += val

        # Normalize
        if self.ret_norm:
            adv = (adv-np.mean(adv))/(np.std(adv) + ret_eps)

        # Clip if required
        if self.ret_clip: adv = np.maximum(adv, 0.0)

        return tgt, adv
