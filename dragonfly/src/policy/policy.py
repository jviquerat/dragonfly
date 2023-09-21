	# Custom imports
from dragonfly.src.core.factory          import *
from dragonfly.src.policy.categorical    import *
from dragonfly.src.policy.normal_diag    import *
from dragonfly.src.policy.normal_full    import *
from dragonfly.src.policy.tanh_normal    import *
from dragonfly.src.policy.beta           import *
from dragonfly.src.policy.deterministic  import *
from dragonfly.src.policy.normal_iso     import *
from dragonfly.src.policy.normal_const   import *


# Declare factory
pol_factory = factory()

# Register policies
pol_factory.register("categorical",    categorical)
pol_factory.register("normal",         normal_diag)
pol_factory.register("normal_diag",    normal_diag)
pol_factory.register("normal_full",    normal_full)
pol_factory.register("tanh_normal",    tanh_normal)
pol_factory.register("beta",           beta)
pol_factory.register("deterministic",  deterministic)
pol_factory.register("normal_iso ",    normal_iso)
pol_factory.register("normal_const",   normal_const)


