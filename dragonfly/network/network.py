# Custom imports
from dragonfly.core.factory  import *
from dragonfly.network.fc    import *

# Declare factory
net_factory = factory()

# Register values
net_factory.register("fc", fc)
