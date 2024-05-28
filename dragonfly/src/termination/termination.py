# Custom imports
from dragonfly.src.core.factory          import factory
from dragonfly.src.termination.bootstrap import bootstrap
from dragonfly.src.termination.regular   import regular

# Declare factory
termination_factory = factory()

# Register values
termination_factory.register("bootstrap", bootstrap)
termination_factory.register("regular",   regular)
