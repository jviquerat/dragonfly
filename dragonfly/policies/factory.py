# Custom imports
from dragonfly.policies.multinomial import *
from dragonfly.policies.normal      import *

###############################################
### A very basic policy factory
class policy_factory:
    def __init__(self):
        self.policies = ["multinomial", "normal"]

    def create(self, key, **kwargs):
        if (key not in self.policies): raise ValueError(key)
        if (key == "multinomial"): return multinomial(**kwargs)
        if (key == "normal"):      return normal(**kwargs)

# Declare factory
policy_factory = policy_factory()
