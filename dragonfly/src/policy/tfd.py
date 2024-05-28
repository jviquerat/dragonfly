# Generic imports
import os
import math
import numpy as np
import warnings

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow             as tf
import tensorflow_probability as tfp

# Define alias
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tfd = tfp.distributions

tf.random.set_seed(0)
