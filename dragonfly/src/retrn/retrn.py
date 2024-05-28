# Custom imports
from dragonfly.src.core.factory    import factory
from dragonfly.src.retrn.full      import full
from dragonfly.src.retrn.advantage import advantage
from dragonfly.src.retrn.gae       import gae
from dragonfly.src.retrn.cgae      import cgae

# Declare factory
retrn_factory = factory()

# Register values
retrn_factory.register("full",      full)
retrn_factory.register("advantage", advantage)
retrn_factory.register("gae",       gae)
retrn_factory.register("cgae",      cgae)

