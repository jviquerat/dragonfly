# Generic imports
import numpy as np

###############################################
### Class for bootstrap terminator
### pms : parameters
class bootstrap():
    def __init__(self, n_cpu, pms):

        # Set values
        self.n_cpu  = n_cpu
        self.ep_end = pms.ep_end

    # Terminate buffers
    def terminate(self, counter, done):

        # Initialize arrays
        trm = np.zeros([self.n_cpu])
        bts = np.zeros([self.n_cpu])

        # Loop over environments
        for i in range(self.n_cpu):

            # Set terminal value, whatever the cause
            trm[i] = float(not (done[i] == True))

            # If bootstrap is on, test and fill
            step = counter.ep_step[i]
            if (step >= self.ep_end-1): bts[i] = 1.0

        return trm, bts
