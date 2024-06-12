# Generic imports
import matplotlib.pyplot as plt

# Custom imports
from dragonfly.src.core.constants import *
from dragonfly.src.utils.buff     import gbuff
from dragonfly.src.utils.counter  import counter

###############################################
### Base srl
class base_srl():
    def __init__(self):
        pass

    def reset(self):
        pass

    def store_obs(self, obs):

        # Store
        if (self.n_update < self.n_update_max):
            self.gbuff.store(["obs"], [obs])
            self.counter += len(obs)

        # If warmup has not been done yet
        if (self.n_update == 0 and
            self.counter  > self.warmup):

            self.update()
            self.n_update += 1
            self.counter   = 0
            return

        # Otherwise
        if (self.n_update > 0 and
            self.n_update < self.n_update_max and
            self.counter  > self.retrain_freq):

            self.update()
            self.n_update += 1
            self.counter   = 0
            return

    def save(self, filename):
        pass

    def load(self, filename):
        pass
