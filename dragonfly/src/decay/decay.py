# Custom imports
from dragonfly.src.core.factory      import factory
from dragonfly.src.decay.linear      import linear
from dragonfly.src.decay.exponential import exponential
from dragonfly.src.decay.sawtooth    import sawtooth

# Declare factory
decay_factory = factory()

# Register values
decay_factory.register("linear",      linear)
decay_factory.register("exponential", exponential)
decay_factory.register("sawtooth",    sawtooth)
