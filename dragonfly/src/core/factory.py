# Custom imports
from dragonfly.src.utils.error import error

###############################################
### A very basic factory
class factory:
    def __init__(self):
        self.keys = {}

    def register(self, key, creator):
        self.keys[key] = creator

    def create(self, key, **kwargs):
        creator = self.keys.get(key)
        if not creator:
            try:
                raise ValueError(key)
            except ValueError:
                error("factory", "create", "Unknown key provided: "+key)
                raise
        return creator(**kwargs)
