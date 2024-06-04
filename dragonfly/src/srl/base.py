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

        self.gbuff.store(["obs"], [obs])
        self.counter += len(obs)

    def save(self, filename):
        pass

    def load(self, filename):
        pass
