# Custom imports
from dragonfly.core.factory  import *
from dragonfly.advantage.gae import *

# Declare factory
advantage_factory = factory()

# Register values
advantage_factory.register("gae", gae)
