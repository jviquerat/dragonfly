# Custom imports
from dragonfly.core.factory    import *
from dragonfly.optimizer.nadam import *

# Declare factory
opt_factory = factory()

# Register values
opt_factory.register("nadam", nadam)
