# Custom imports
from dragonfly.core.factory       import *
from dragonfly.policy.multinomial import *
from dragonfly.policy.normal      import *

# Declare factory
pol_factory = factory()

# Register policies
pol_factory.register("multinomial", multinomial)
pol_factory.register("normal",      normal)

