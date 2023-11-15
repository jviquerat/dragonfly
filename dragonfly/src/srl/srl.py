# Custom imports
from dragonfly.src.core.factory import *
from dragonfly.src.srl.dummy    import *
from dragonfly.src.srl.pca      import *
from dragonfly.src.srl.ae      import *
from dragonfly.src.srl.kpca     import *

# Declare factory
srl_factory = factory()

# Register srl
srl_factory.register("dummy", dummy)
srl_factory.register("pca", pca)
srl_factory.register("ae", ae)
srl_factory.register("kpca", kpca)
