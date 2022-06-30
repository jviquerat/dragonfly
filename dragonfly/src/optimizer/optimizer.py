# Custom imports
from dragonfly.src.core.factory       import *
from dragonfly.src.optimizer.nadam    import *
from dragonfly.src.optimizer.adam     import *
from dragonfly.src.optimizer.adadelta import *
from dragonfly.src.optimizer.adamax   import *
from dragonfly.src.optimizer.sgd      import *

# Declare factory
opt_factory = factory()

# Register values
opt_factory.register("nadam",    nadam)
opt_factory.register("adam",     adam)
opt_factory.register("adadelta", adadelta)
opt_factory.register("adamax",   adamax)
opt_factory.register("sgd",      sgd)
