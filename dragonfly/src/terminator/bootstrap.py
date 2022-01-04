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
    def terminate(self, buff):

        # Initialize
        trm = np.zeros([self.n_cpu])
        bts = np.zeros([self.n_cpu])

        # Loop over steps in buffer
        for i in range(buff.length()):

            trm[:] = 0.0
            bts[:] = 0.0

            # Loop over parallel environments
            for j in range(self.n_cpu):

                # Set terminal value, whatever the cause
                trm[j] = float(not (buff.dne.buff[j][i] == 1.0))

                # If bootstrap is on, test and fill
                if (buff.stp.buff[j][i] >= self.ep_end-1): bts[j] = 1.0

            # Store terminal arrays
            print(trm)
            print(bts)
            print("")
            buff.store_terminal(trm, bts)

        # When using buffer-based updates, the last step of each
        # buffer must be bootstraped to mimic a continuing episode
        for j in range(self.n_cpu):
            if (buff.trm.buff[j][-1] == 1.0):
                buff.bts.buff[j][-1] =  1.0
                buff.trm.buff[j][-1] =  0.0
