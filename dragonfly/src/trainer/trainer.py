# Custom imports
from dragonfly.src.core.factory         import *
from dragonfly.src.trainer.buffer_based import *

# Declare factory
trainer_factory = factory()

# Register trainers
trainer_factory.register("buffer_based", buffer_based)
