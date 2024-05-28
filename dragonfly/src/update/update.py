# Custom imports
from dragonfly.src.core.factory      import factory
from dragonfly.src.update.on_policy  import on_policy
from dragonfly.src.update.off_policy import off_policy

# Declare factory
update_factory = factory()

# Register updates
update_factory.register("on_policy",  on_policy)
update_factory.register("off_policy", off_policy)
