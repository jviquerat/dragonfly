# Custom imports
from dragonfly.src.core.factory   import *
from dragonfly.src.loss.mse_pg    import *
from dragonfly.src.loss.mse_dqn   import *
from dragonfly.src.loss.mse_ddpg  import *
from dragonfly.src.loss.mse_td3   import *
from dragonfly.src.loss.mse_sac   import *
from dragonfly.src.loss.mse_ae    import *
from dragonfly.src.loss.surrogate import *
from dragonfly.src.loss.pg        import *
from dragonfly.src.loss.q_pol     import *
from dragonfly.src.loss.q_pol_sac import *

# Declare factory
loss_factory = factory()

# Register values
loss_factory.register("mse_pg",    mse_pg)
loss_factory.register("mse_dqn",   mse_dqn)
loss_factory.register("mse_ddpg",  mse_ddpg)
loss_factory.register("mse_td3",   mse_td3)
loss_factory.register("mse_sac",   mse_sac)
loss_factory.register("mse_ae",    mse_ae)
loss_factory.register("surrogate", surrogate)
loss_factory.register("pg",        pg)
loss_factory.register("q_pol",     q_pol)
loss_factory.register("q_pol_sac", q_pol_sac)
