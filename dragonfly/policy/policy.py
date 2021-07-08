# Custom imports
from dragonfly.core.factory       import *
from dragonfly.policy.categorical import *
from dragonfly.policy.normal      import *

# Declare factory
pol_factory = factory()

# Register policies
pol_factory.register("categorical", categorical)
pol_factory.register("normal",      normal)

