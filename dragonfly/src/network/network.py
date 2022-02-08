# Custom imports
from dragonfly.src.core.factory  import *
from dragonfly.src.network.fc    import *

# Declare factory
net_factory = factory()

# Register values
net_factory.register("fc", fc)
