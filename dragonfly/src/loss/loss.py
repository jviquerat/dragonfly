# Custom imports
from dragonfly.src.core.factory   import *
from dragonfly.src.loss.mse       import *
from dragonfly.src.loss.mse_dqn   import *
from dragonfly.src.loss.surrogate import *
from dragonfly.src.loss.pg        import *

# Declare factory
loss_factory = factory()

# Register values
loss_factory.register("mse",       mse)
loss_factory.register("mse_dqn",   mse_dqn)
loss_factory.register("surrogate", surrogate)
loss_factory.register("pg",        pg)
