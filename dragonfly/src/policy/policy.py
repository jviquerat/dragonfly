# Custom imports
from dragonfly.src.core.factory          import factory
from dragonfly.src.policy.categorical    import categorical
from dragonfly.src.policy.normal_const   import normal_const
from dragonfly.src.policy.normal_iso     import normal_iso
from dragonfly.src.policy.normal_diag    import normal_diag
from dragonfly.src.policy.normal_full    import normal_full
from dragonfly.src.policy.tanh_normal    import tanh_normal
from dragonfly.src.policy.deterministic  import deterministic

# Declare factory
pol_factory = factory()

# Register policies
pol_factory.register("categorical",    categorical)
pol_factory.register("normal",         normal_iso)
pol_factory.register("normal_const",   normal_const)
pol_factory.register("normal_iso",     normal_iso)
pol_factory.register("normal_diag",    normal_diag)
pol_factory.register("normal_full",    normal_full)
pol_factory.register("tanh_normal",    tanh_normal)
pol_factory.register("deterministic",  deterministic)
