# Custom imports
from dragonfly.src.core.factory import *
from dragonfly.src.srl.dummy    import *

# Declare factory
srl_factory = factory()

# Register srl
srl_factory.register("dummy", dummy)
