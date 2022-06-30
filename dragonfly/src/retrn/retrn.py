# Custom imports
from dragonfly.src.core.factory    import *
from dragonfly.src.retrn.full      import *
from dragonfly.src.retrn.advantage import *
from dragonfly.src.retrn.gae       import *

# Declare factory
retrn_factory = factory()

# Register values
retrn_factory.register("full",      full)
retrn_factory.register("advantage", advantage)
retrn_factory.register("gae",       gae)

