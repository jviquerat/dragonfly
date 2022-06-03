# Generic imports
import numpy as np

###############################################
### Class for sawtooth decay
### pms : parameters
class sawtooth():
    def __init__(self, pms):

        # Set values
        self.start     = pms.start
        self.end       = pms.end
        self.period    = pms.period
        self.n_periods = pms.n_periods

        # Initialize return values
        self.step     = 0
        self.step_tot = 0
        self.val      = self.start

    # Get current value
    def get(self):

        return self.val

    # Decay
    def decay(self):

        if (self.step_tot < self.n_periods*self.period):
            r        = float(self.step/self.period)
            self.val = self.start + r*(self.end-self.start)

            self.step     += 1
            self.step_tot += 1

            if (self.step == self.period):
                self.step = 0
        else:
            self.val = self.end
