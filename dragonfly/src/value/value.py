# Custom imports
from dragonfly.src.core.factory  import *
from dragonfly.src.value.v_value import *

# Declare factory
val_factory = factory()

# Register values
val_factory.register("v_value", v_value)
