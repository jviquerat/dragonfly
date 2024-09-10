# Custom imports
from dragonfly.src.core.factory import factory
from dragonfly.src.network.fc import fc
from dragonfly.src.network.conv1d import conv1d
from dragonfly.src.network.conv2d import conv2d

# Declare factory
net_factory = factory()

# Register values
net_factory.register("fc", fc)
net_factory.register("conv1d", conv1d)
net_factory.register("conv2d", conv2d)
