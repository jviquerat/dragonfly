# Custom imports
from dragonfly.src.core.factory      import factory
from dragonfly.src.trainer.buffer    import buffer
from dragonfly.src.trainer.episode   import episode
from dragonfly.src.trainer.td        import td
from dragonfly.src.trainer.separable import separable

# Declare factory
trainer_factory = factory()

# Register trainers
trainer_factory.register("buffer",    buffer)
trainer_factory.register("episode",   episode)
trainer_factory.register("td",        td)
trainer_factory.register("separable", separable)
