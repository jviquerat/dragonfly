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
from   tensorflow.keras.initializers import Orthogonal

###############################################
### Network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### arch     : architecture of densely connected network
### hid_init : hidden layer kernel initializer
### fnl_init : final  layer kernel initializer
### hid_act  : hidden layer activation function
### fnl_act  : final  layer activation function
class network(Model):
    def __init__(self, inp_dim, out_dim, arch,
                 hid_init, fnl_init, hid_act, fnl_act):
        super(network, self).__init__()

        # Initialize network
        self.net = []

        # Define hidden layers
        for layer in range(len(arch)):
            self.net.append(Dense(arch[layer],
                                  kernel_initializer = hid_init,
                                  activation         = hid_act))
        # Define last layer
        self.net.append(Dense(out_dim,
                              kernel_initializer = fnl_init,
                              activation         = fnl_act))

        # Initialize weights
        dummy = self.call(tf.ones([1,inp_dim]))

        # Save initial weights
        self.init_weights = self.get_weights()

    # Network forward pass
    def call(self, var):

        # Compute output
        for layer in range(len(self.net)):
            var = self.net[layer](var)

        return var

    # Reset weights
    def reset(self):

        self.set_weights(self.init_weights)
