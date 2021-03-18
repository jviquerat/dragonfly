# Custom imports
from dragonfly.policy.multinomial import *
from dragonfly.policy.normal      import *

###############################################
### A very basic factory
class policy_factory:
    def __init__(self):
        self.policies = ["multinomial", "normal"]

    def create(self, key, **kwargs):
        if (key not in self.policies): raise ValueError(key)
        if (key == "multinomial"): return multinomial(**kwargs)
        if (key == "normal"):      return normal(**kwargs)

# Declare factory
policy_factory = policy_factory()
