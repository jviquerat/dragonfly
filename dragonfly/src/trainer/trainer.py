# Custom imports
from dragonfly.src.core.factory    import *
from dragonfly.src.trainer.buffer  import *
from dragonfly.src.trainer.episode import *
from dragonfly.src.trainer.td      import *

# Declare factory
trainer_factory = factory()

# Register trainers
trainer_factory.register("buffer",  buffer)
trainer_factory.register("episode", episode)
trainer_factory.register("td",      td)
