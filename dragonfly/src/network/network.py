# Custom imports
from dragonfly.src.core.factory   import *
from dragonfly.src.network.fc     import *
from dragonfly.src.network.d2rl   import *
from dragonfly.src.network.conv1d import *
from dragonfly.src.network.lstm   import *

# Declare factory
net_factory = factory()

# Register values
net_factory.register("fc",     fc)
net_factory.register("d2rl",   d2rl)
net_factory.register("conv1d", conv1d)
net_factory.register("lstm",   lstm)
