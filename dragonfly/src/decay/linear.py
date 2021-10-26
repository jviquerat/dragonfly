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
        self.n_steps  = pms.n_steps

        # Initialize return values
        self.step     = 0
        self.val      = self.start

    # Get current value and decay
    def get(self):

        # Check if max step number is reached
        if (self.step <= self.n_steps):
            r          = float(self.step/self.n_steps)
            self.val   = self.start + r*(self.end-self.start)
            self.step += 1

        return self.val
