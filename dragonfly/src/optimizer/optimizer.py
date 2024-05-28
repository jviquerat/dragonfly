# Custom imports
from dragonfly.src.core.factory    import factory
from dragonfly.src.optimizer.nadam import nadam
from dragonfly.src.optimizer.adam  import adam
from dragonfly.src.optimizer.sgd   import sgd

# Declare factory
opt_factory = factory()

# Register values
opt_factory.register("nadam", nadam)
opt_factory.register("adam",  adam)
opt_factory.register("sgd",   sgd)
