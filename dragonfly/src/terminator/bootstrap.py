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
        epn = np.zeros([self.n_cpu])

        # Loop over environments
        for i in range(self.n_cpu):

            # Set terminal value, whatever the cause
            trm[i] = float(not (done[i] == True))

            # If bootstrap is on, test and fill
            step = counter.ep_step[i]
            if (step >= self.ep_end-1): bts[i] = 1.0

            # Store episode number for each step
            epn[i] = counter.ep

        return trm, bts, epn

    # Bootstrap terminal step of buffer once buffer is full
    def bootstrap_terminal(self, loc_buff):

        # When using buffer-based updates, the last step of each
        # buffer must be bootstraped to mimic a continuing episode
        for cpu in range(self.n_cpu):
            if (loc_buff.trm.buff[cpu][-1] == 1.0):
                loc_buff.bts.buff[cpu][-1] = 1.0
                loc_buff.trm.buff[cpu][-1] = 0.0
