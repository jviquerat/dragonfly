# Custom imports
from dragonfly.agents.ppo import *

###############################################
### A very basic agent factory
class agent_factory:
    def __init__(self):
        self.agents = ["ppo"]

    def create(self, key, **kwargs):
        if (key not in self.agents): raise ValueError(key)
        if (key == "ppo"): return ppo(**kwargs)

# Declare factory
agent_factory = agent_factory()
