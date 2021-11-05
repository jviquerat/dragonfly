# Custom imports
from dragonfly.src.core.factory import *
from dragonfly.src.terminator.bootstrap import *

# Declare factory
terminator_factory = factory()

# Register values
terminator_factory.register("bootstrap", bootstrap)
