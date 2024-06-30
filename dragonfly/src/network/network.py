# Custom imports
from dragonfly.src.core.factory      import factory
from dragonfly.src.network.fc        import fc
from dragonfly.src.network.conv1d    import conv1d
from dragonfly.src.network.conv2d    import conv2d
from dragonfly.src.network.lstm      import LSTM
from dragonfly.src.network.gated_fc  import GatedFC
from dragonfly.src.network.ae        import ae
from dragonfly.src.network.conv2d_ae import *

# Declare factory
net_factory = factory()

# Register values
net_factory.register("fc",        fc)
net_factory.register("conv1d",    conv1d)
net_factory.register("conv2d",    conv2d)
net_factory.register("lstm",      LSTM)
net_factory.register("gated_fc",  GatedFC)
net_factory.register("ae",        ae)
net_factory.register("conv2d_ae", conv2d_ae)
