# Custom imports
from dragonfly.core.factory  import *
from dragonfly.advantage.gae import *

# Declare factory
adv_factory = factory()

# Register values
adv_factory.register("gae", gae)
