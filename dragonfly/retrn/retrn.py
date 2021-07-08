# Custom imports
from dragonfly.core.factory      import *
from dragonfly.retrn.gae         import *
from dragonfly.retrn.advantage   import *
from dragonfly.retrn.full_return import *

# Declare factory
retrn_factory = factory()

# Register values
retrn_factory.register("gae",         gae)
retrn_factory.register("advantage",   advantage)
retrn_factory.register("full_return", full_return)
