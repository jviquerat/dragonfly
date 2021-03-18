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
### Fully-connected network class
### inp_dim  : dimension of input  layer
### out_dim  : dimension of output layer
### arch     : architecture of densely connected network
### hid_init : hidden layer kernel initializer
### fnl_init : final  layer kernel initializer
### hid_act  : hidden layer activation function
### fnl_act  : final  layer activation function
class fc(Model):
    def __init__(self, inp_dim, out_dim, pms):
        super().__init__()

        # Set inputs
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Set default values
        self.arch     = [32,32]
        self.hid_init = Orthogonal(gain=1.0)
        self.fnl_init = Orthogonal(gain=0.01)
        self.hid_actv = "tanh"
        self.fnl_actv = "linear"

        # Check inputs
        if hasattr(pms, "arch"):     self.arch     = pms.arch
        if hasattr(pms, "hid_init"): self.hid_init = pms.hid_init
        if hasattr(pms, "fnl_init"): self.fnl_init = pms.fnl_init
        if hasattr(pms, "hid_actv"): self.hid_actv = pms.hid_actv
        if hasattr(pms, "fnl_actv"): self.fnl_actv = pms.fnl_actv

        # Initialize network
        self.net = []

        # Define hidden layers
        for layer in range(len(self.arch)):
            self.net.append(Dense(self.arch[layer],
                                  kernel_initializer = self.hid_init,
                                  activation         = self.hid_actv))
        # Define last layer
        self.net.append(Dense(self.out_dim,
                              kernel_initializer = self.fnl_init,
                              activation         = self.fnl_actv))

        # Initialize weights
        dummy = self.call(tf.ones([1,self.inp_dim]))

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
