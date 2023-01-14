# Generic imports
import os
import warnings

# Filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)

# Tensorflow imports
import tensorflow                    as     tf
import tensorflow.keras              as     tk
from   tensorflow.keras              import Model
from   tensorflow.keras.layers       import Dense
from   tensorflow.keras.initializers import Orthogonal, LecunNormal, Zeros

# Custom imports
from   dragonfly.src.utils.error     import *
from   dragonfly.src.network.tree    import *

###############################################
### Base network
class base_network(Model):
    def __init__(self):
        super().__init__()

    # Network forward pass
    def call(self, x):
        raise NotImplementedError

    # Reset weights
    def reset(self):
        raise NotImplementedError
