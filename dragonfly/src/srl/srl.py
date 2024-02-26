# Custom imports
from dragonfly.src.core.factory import *
from dragonfly.src.srl.dummy    import *
from dragonfly.src.srl.pca      import *
from dragonfly.src.srl.sae      import *
from dragonfly.src.srl.vae      import *
from dragonfly.src.srl.rae      import *
from dragonfly.src.srl.kpca     import *
from dragonfly.src.srl.aeConv1D     import *

# Declare factory
srl_factory = factory()

# Register srl
srl_factory.register("dummy", dummy)
srl_factory.register("pca",   pca)
srl_factory.register("sae",   sae)
srl_factory.register("vae",   vae)
srl_factory.register("rae",   rae)
srl_factory.register("kpca",  kpca)
srl_factory.register("aeConv1D",  aeConv1D)
