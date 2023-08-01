# Custom imports
from dragonfly.src.core.factory   import *
from dragonfly.src.update.online  import *
from dragonfly.src.update.offline import *
from dragonfly.src.update.cma     import *

# Declare factory
update_factory = factory()

# Register updates
update_factory.register("online",  online)
update_factory.register("offline", offline)
update_factory.register("cma",     cma)
