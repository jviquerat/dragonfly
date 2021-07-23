# Custom imports
from dragonfly.src.core.factory import *
from dragonfly.src.loss.mse     import *
from dragonfly.src.loss.ppo     import *
from dragonfly.src.loss.pg      import *

# Declare factory
loss_factory = factory()

# Register values
loss_factory.register("mse", mse)
loss_factory.register("ppo", ppo)
loss_factory.register("pg",  pg)
