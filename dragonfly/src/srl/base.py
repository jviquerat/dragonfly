# Generic imports
import random
import numpy as np

# Custom imports
from dragonfly.src.core.constants import *
from dragonfly.src.utils.buff     import *
from dragonfly.src.utils.counter  import *

###############################################
### Base srl
class base_srl():
    def __init__(self):
        pass

    def reset(self):
        pass

    def store(self, name, x):

        self.gbuff.store([name], x)

    def update_counter(self):

        self.counter += 1
