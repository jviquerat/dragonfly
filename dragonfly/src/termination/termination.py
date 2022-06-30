# Custom imports
from dragonfly.src.core.factory          import *
from dragonfly.src.termination.bootstrap import *
from dragonfly.src.termination.regular   import *

# Declare factory
termination_factory = factory()

# Register values
termination_factory.register("bootstrap", bootstrap)
termination_factory.register("regular",   regular)
