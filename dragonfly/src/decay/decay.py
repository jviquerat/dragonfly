# Custom imports
from dragonfly.src.core.factory import *
from dragonfly.src.decay.linear import *

# Declare factory
decay_factory = factory()

# Register values
decay_factory.register("linear", linear)
