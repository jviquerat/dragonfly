# Custom imports
from dragonfly.value.v_value import *

###############################################
### A very basic value factory
class value_factory:
    def __init__(self):
        self.values = ["v_value"]

    def create(self, key, **kwargs):
        if (key not in self.values): raise ValueError(key)
        if (key == "v_value"): return v_value(**kwargs)

# Declare factory
value_factory = value_factory()
