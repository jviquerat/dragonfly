# Generic imports
import numpy as np

###############################################
### Class for regular terminator
### pms : parameters
class regular():
    def __init__(self, n_cpu, pms):

        # Set values
        self.n_cpu = n_cpu

    # Terminate buffers
    def terminate(self, buff):

        # Initialize arrays
        trm = np.zeros([self.n_cpu])

        # Loop over steps
        for k in range(buff.length()):

            trm[:] = 0.0

            # Loop over environments
            for i in range(self.n_cpu):

                # Set terminal value, whatever the cause
                trm[i] = float(not (buff.data["dne"].buff[i][k] == True))

            # Store terminal data to buffer
            buff.store(["trm"], [trm])
