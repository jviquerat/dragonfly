# Custom imports
from dragonfly.src.core.factory      import *
from dragonfly.src.retrn.gae         import *
from dragonfly.src.retrn.advantage   import *
from dragonfly.src.retrn.full_return import *

# Declare factory
retrn_factory = factory()

# Register values
retrn_factory.register("gae",         gae)
retrn_factory.register("advantage",   advantage)
retrn_factory.register("full_return", full_return)
