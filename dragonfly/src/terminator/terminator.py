# Custom imports
from dragonfly.src.core.factory         import *
from dragonfly.src.terminator.bootstrap import *
from dragonfly.src.terminator.regular   import *

# Declare factory
terminator_factory = factory()

# Register values
terminator_factory.register("bootstrap", bootstrap)
terminator_factory.register("regular",   regular)
