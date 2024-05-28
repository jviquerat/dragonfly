# Custom imports
from dragonfly.src.core.factory  import factory
from dragonfly.src.value.v_value import v_value
from dragonfly.src.value.q_value import q_value

# Declare factory
val_factory = factory()

# Register values
val_factory.register("v_value", v_value)
val_factory.register("q_value", q_value)
