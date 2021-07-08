# Custom imports
from dragonfly.core.factory import *
from dragonfly.loss.mse     import *
from dragonfly.loss.ppo     import *
from dragonfly.loss.pg      import *

# Declare factory
loss_factory = factory()

# Register values
loss_factory.register("mse", mse)
loss_factory.register("ppo", ppo)
loss_factory.register("pg",  pg)
