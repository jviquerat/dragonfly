# Generic imports
import numpy as np

# Custom imports
from dragonfly.core.network import *

###############################################
### Critic class
### obs_dim  : dimension of input  layer
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
                 obs_dim,
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

        # Fill structure
        self.loss    = loss
        self.obs_dim = obs_dim

        # Define network
        self.net  = network(obs_dim, dim, arch, lr, grd_clip,
                            hid_init, fnl_init, hid_act, fnl_act)

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

    # Get value from network
    def get_value(self, state):

        # Reshape state
        state = tf.cast(state, tf.float32)

        # Predict value
        val   = np.array(self.call(state))

        return val

    # Get current learning rate
    def get_lr(self):

        return self.opt._decayed_lr(tf.float32)

    # MSE loss function for critic
    @tf.function
    def train(self, obs, tgt, btc):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.convert_to_tensor(self.call(obs))
            val  = tf.reshape(val, [btc])
            p1   = tf.square(tgt - val)
            loss = tf.reduce_mean(p1)

            # Apply gradients
            crt_var     = self.net.trainable_variables
            grads       = tape.gradient(loss, crt_var)
            norm        = tf.linalg.global_norm(grads)
        self.opt.apply_gradients(zip(grads,crt_var))

        return loss, norm
