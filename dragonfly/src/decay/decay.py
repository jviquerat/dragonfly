# Custom imports
from dragonfly.src.core.factory      import *
from dragonfly.src.decay.linear      import *
from dragonfly.src.decay.exponential import *
from dragonfly.src.decay.sawtooth    import *

# Declare factory
decay_factory = factory()

# Register values
decay_factory.register("linear",      linear)
decay_factory.register("exponential", exponential)
decay_factory.register("sawtooth",    sawtooth)
