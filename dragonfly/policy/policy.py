# Custom imports
from dragonfly.core.factory       import *
from dragonfly.policy.multinomial import *
from dragonfly.policy.normal      import *

# Declare factory
policy_factory = factory()

# Register policies
policy_factory.register("multinomial", multinomial)
policy_factory.register("normal",      normal)

