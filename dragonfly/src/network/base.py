# Generic imports
import os
import warnings

# Filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)

# Tensorflow imports
import tensorflow                    as     tf
from   tensorflow.keras              import Model
from   tensorflow.keras.layers       import Dense, Flatten, Reshape
from   tensorflow.keras.initializers import Orthogonal, LecunNormal

# Custom imports
from   dragonfly.src.utils.error     import error, warning
from   dragonfly.src.network.tree    import trunk, heads

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

    # Return trainable parameters
    def trainables(self):

        return self.trainable_weights
