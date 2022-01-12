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

        # Initialize arrays
        trm = np.zeros([self.n_cpu])
        bts = np.zeros([self.n_cpu])

        # Loop over steps
        for k in range(buff.length()):

            trm[:] = 0.0
            bts[:] = 0.0

            # Loop over environments
            for i in range(self.n_cpu):

                # Set terminal value, whatever the cause
                trm[i] = float(not (buff.data["dne"].buff[i][k] == True))

                # Bootstrap end of episode if it is a regular ending
                if (buff.data["stp"].buff[i][k] >= self.ep_end-1): bts[i] = 1.0

            # Store terminal data to buffer
            buff.store(["trm", "bts"], [trm, bts])

        # When using buffer-based updates, the last step of each
        # buffer must be bootstraped to mimic a continuing episode
        # This operation doesn't affect episode-based training[O]
        for i in range(self.n_cpu):
            if (buff.data["trm"].buff[i][-1] == 1.0):
                buff.data["bts"].buff[i][-1] =  1.0
                buff.data["trm"].buff[i][-1] =  0.0
