# Custom imports
from dragonfly.src.core.factory  import *
from dragonfly.src.network.fc    import *
from dragonfly.src.network.ae    import *
from dragonfly.src.network.vae   import *
from dragonfly.src.network.rae   import *
from dragonfly.src.network.snn   import *
from dragonfly.src.network.aeConv1D   import *

# Declare factory
net_factory = factory()

# Register values
net_factory.register("fc",  fc)
net_factory.register("ae",  ae)
net_factory.register("vae", vae)
net_factory.register("rae", rae)
net_factory.register("snn", snn)
net_factory.register("aeConv1D", aeConv1D)
