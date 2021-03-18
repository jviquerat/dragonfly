# Custom imports
from dragonfly.advantage.gae import *

###############################################
### A very basic factory
class advantage_factory:
    def __init__(self):
        self.advantages = ["full_return", "gae"]

    def create(self, key, **kwargs):
        if (key not in self.advantages): raise ValueError(key)
        if (key == "full_return"): return full_return(**kwargs)
        if (key == "gae"):         return gae(**kwargs)

# Declare factory
advantage_factory = advantage_factory()
