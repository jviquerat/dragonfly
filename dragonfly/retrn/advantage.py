# Generic imports
import numpy as np

###############################################
### Class for advantage
### pms : parameters
class advantage():
    def __init__(self, pms):

        # Set default values
        self.gamma      = 0.99
        self.ret_norm   = True
        self.ret_clip   = True

        # Check inputs
        if hasattr(pms, "gamma"):      self.gamma      = pms.gamma
        if hasattr(pms, "ret_norm"):   self.ret_norm   = pms.ret_norm
        if hasattr(pms, "ret_clip"):   self.ret_clip   = pms.ret_clip

    # Compute advantage
    # rwd : reward array
    # val : value array
    # nxt : value array shifted by one timestep
    # trm : array of terminal values
    def compute(self, rwd, val, nxt, trm):

        # Shortcuts
        gm  = self.gamma

        # Handle mask from termination signals
        msk = np.zeros(len(trm))
        for i in range(len(trm)):
            if (trm[i] == 0): msk[i] = 1.0
            if (trm[i] == 1): msk[i] = 0.0
            if (trm[i] == 2): msk[i] = 1.0

        # Initialize return term==2
        ret = np.where(trm == 2, rwd + gm * nxt, rwd)

        # Return as discounted sum
        for t in reversed(range(len(ret)-1)):
            ret[t] += msk[t]*gm*ret[t+1]

        # Advantage
        adv = ret - val

        # Compute targets
        tgt = ret.copy()

        # Normalize
        if self.ret_norm: adv = (adv-np.mean(adv))/(np.std(adv) + 1.0e-5)

        # Clip if required
        if self.ret_clip: adv = np.maximum(adv, 0.0)

        return tgt, adv
