# Custom imports
from dragonfly.src.core.factory    import factory
from dragonfly.src.optimizer.adam  import adam
from dragonfly.src.optimizer.adamw import adamw
from dragonfly.src.optimizer.sgd   import sgd

# Declare factory
opt_factory = factory()

# Register values
opt_factory.register("adam",  adam)
opt_factory.register("adamw", adamw)
opt_factory.register("sgd",   sgd)
