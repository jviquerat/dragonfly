# Custom imports
from dragonfly.src.core.factory       import *
from dragonfly.src.policy.categorical import *
from dragonfly.src.policy.normal      import *
from dragonfly.src.policy.beta        import *

# Declare factory
pol_factory = factory()

# Register policies
pol_factory.register("categorical", categorical)
pol_factory.register("normal",      normal)
pol_factory.register("beta",        beta)

