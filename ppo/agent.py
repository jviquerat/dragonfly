# Generic imports
from tensorflow.keras.optimizers import Nadam

# Custom imports
from ppo.network                 import *

###############################################
### Actor class
### dim      : dimension of output layer
### arch     : architecture of densely connected network
### lr       : learning rate
### grd_clip : gradient clipping value
### hid_init : hidden layer kernel initializer
### fnl_init : final  layer kernel initializer
### hid_act  : hidden layer activation function
### fnl_act  : final  layer activation function
### loss     : loss function
class actor():
    def __init__(self,
                 dim,
                 arch     = None,
                 lr       = None,
                 grd_clip = None,
                 hid_init = None,
                 fnl_init = None,
                 hid_act  = None,
                 fnl_act  = None,
                 loss     = None):

        # Handle arguments
        if (arch     is None): arch     = [32,32]
        if (lr       is None): lr       = 1.0e-3
        if (grd_clip is None): grd_clip = 0.1
        if (hid_init is None): hid_init = Orthogonal(gain=1.0)
        if (fnl_init is None): fnl_init = Orthogonal(gain=0.01)
        if (hid_act  is None): hid_act  = "tanh"
        if (fnl_act  is None): fnl_act  = "softmax"
        if (loss     is None): loss     = "ppo"

        # Define network
        self.net = network(dim, arch, lr, grd_clip,
                           hid_init, fnl_init, hid_act, fnl_act)
        self.loss = loss

        # Define optimizer
        self.opt = Nadam(lr       = lr,
                         clipnorm = grd_clip)

    # Network forward pass
    def call(self, state):

        # Copy inputs
        var = state

        # Call network
        var = self.net.call(var)

        return var



    # Get actions
    #def get_actions(self, state):



###############################################
### Critic class
### arch     : architecture of densely connected network
### lr       : learning rate
### grd_clip : gradient clipping value
### hid_init : hidden layer kernel initializer
### fnl_init : final  layer kernel initializer
### hid_act  : hidden layer activation function
### fnl_act  : final  layer activation function
### loss     : loss function
class critic():
    def __init__(self,
                 dim      = None,
                 arch     = None,
                 lr       = None,
                 grd_clip = None,
                 hid_init = None,
                 fnl_init = None,
                 hid_act  = None,
                 fnl_act  = None,
                 loss     = None):

        # Handle arguments
        if (dim      is None): dim      = 1
        if (arch     is None): arch     = [32,32]
        if (lr       is None): lr       = 1.0e-3
        if (grd_clip is None): grd_clip = 1.0e10
        if (hid_init is None): hid_init = Orthogonal(gain=1.0)
        if (fnl_init is None): fnl_init = Orthogonal(gain=1.0)
        if (hid_act  is None): hid_act  = "tanh"
        if (fnl_act  is None): fnl_act  = "linear"
        if (loss     is None): loss     = "mse"

        # Define network
        self.net  = network(dim, arch, lr, grd_clip,
                            hid_init, fnl_init, hid_act, fnl_act)
        self.loss = loss

        # Define optimizer
        self.opt = Nadam(lr       = lr,
                         clipnorm = grd_clip)

    # Network forward pass
    def call(self, state):

        # Copy inputs
        var = state

        # Call network
        var = self.net.call(var)

        return var

    # Get value
    #def get_value(self, state):
