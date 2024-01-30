# Custom imports
from dragonfly.src.core.factory  import *
from dragonfly.src.network.fc    import *
from dragonfly.src.network.ae    import *
from dragonfly.src.network.snn   import *

# Declare factory
net_factory = factory()

# Register values
net_factory.register("fc",  fc)
net_factory.register("ae",  ae)
net_factory.register("snn", snn)
