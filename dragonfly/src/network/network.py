# Custom imports
from dragonfly.src.core.factory   import factory
from dragonfly.src.network.fc     import fc
from dragonfly.src.network.d2rl   import d2rl
from dragonfly.src.network.conv2d import conv2d

# Declare factory
net_factory = factory()

# Register values
net_factory.register("fc",     fc)
net_factory.register("d2rl",   d2rl)
net_factory.register("conv2d", conv2d)
