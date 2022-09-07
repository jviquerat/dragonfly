# Generic imports
import numpy as np

###############################################
### Class for exponential decay
### pms : parameters
class exponential():
    def __init__(self, pms):

        # Set values
        self.start    = pms.start
        self.end      = pms.end
        self.dcy      = pms.decay

        # Initialize return values
        self.reset()

    # Get current value
    def get(self):

        return self.val

    # Decay
    def decay(self):

        self.val = max(self.end, self.val*self.dcy)

    # Reset
    def reset(self):

        self.val = self.start
