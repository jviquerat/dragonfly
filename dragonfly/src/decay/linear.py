# Generic imports
import numpy as np

###############################################
### Class for linear decay
### pms : parameters
class linear():
    def __init__(self, pms):

        # Set values
        self.start    = pms.start
        self.end      = pms.end
        self.n_decay  = pms.n_decay

        # Initialize return values
        self.reset()

    # Get current value
    def get(self):

        return self.val

    # Decay
    def decay(self):

        r          = min(float(self.step/self.n_decay), 1.0)
        self.val   = self.start + r*(self.end-self.start)
        self.step += 1

    # Reset
    def reset(self):

        self.step = 0
        self.val  = self.start
