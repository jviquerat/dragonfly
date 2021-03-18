# Custom imports
from dragonfly.core.factory  import *
from dragonfly.value.v_value import *

# Declare factory
value_factory = factory()

# Register values
value_factory.register("v_value", v_value)
