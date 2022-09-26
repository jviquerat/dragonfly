# Generic imports
import numpy as np

###############################################
### Class for bootstrap termination
### pms : parameters
class bootstrap():
    def __init__(self, n_cpu, pms):

        # Set values
        self.type  = pms.type
        self.n_cpu = n_cpu

    # Terminate buffers
    # trm = 0 if terminal and episode not over
    # trm = 1 if not terminal
    # trm = 2 if terminal and episode over (for bootstrapping)
    def terminate(self, dne, trc):

        trm = np.zeros([self.n_cpu])

        for k in range(self.n_cpu):
            trm[k] = float(not dne[k])
            if trc[k]: trm[k] = 2.0

        return trm
