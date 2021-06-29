# Custom imports
from dragonfly.core.factory       import *
from dragonfly.optimizer.nadam    import *
from dragonfly.optimizer.adam     import *
from dragonfly.optimizer.adadelta import *
from dragonfly.optimizer.sgd      import *

# Declare factory
opt_factory = factory()

# Register values
opt_factory.register("nadam",    nadam)
opt_factory.register("adam",     adam)
opt_factory.register("adadelta", adadelta)
opt_factory.register("sgd",      sgd)
