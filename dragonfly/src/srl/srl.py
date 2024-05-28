# Custom imports
from dragonfly.src.core.factory import factory
from dragonfly.src.srl.dummy    import dummy
from dragonfly.src.srl.pca      import pca
from dragonfly.src.srl.sae      import sae

# Declare factory
srl_factory = factory()

# Register srl
srl_factory.register("dummy", dummy)
srl_factory.register("pca",   pca)
srl_factory.register("sae",   sae)
