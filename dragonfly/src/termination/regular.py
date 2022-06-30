# Generic imports
import numpy as np

###############################################
### Class for regular termination
### pms : parameters
class regular():
    def __init__(self, n_cpu, pms):

        self.type  = pms.type
        self.n_cpu = n_cpu

    # Terminate buffers
    def terminate(self, dne, stp):

        trm = np.zeros([self.n_cpu])

        for k in range(self.n_cpu):
            trm[k] = float(not dne[k])

        return trm
