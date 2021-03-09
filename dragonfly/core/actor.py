# Generic imports
import numpy as np

# Custom imports
from dragonfly.core.network import *
from dragonfly.core.policy  import *

###############################################
### Actor class
### act_dim  : dimension of output layer
### obs_dim  : dimension of input  layer
### arch     : architecture of densely connected network
### lr       : learning rate
### grd_clip : gradient clipping value
### hid_init : hidden layer kernel initializer
### fnl_init : final  layer kernel initializer
### hid_act  : hidden layer activation function
### fnl_act  : final  layer activation function
### loss     : loss function
### policy   : output policy
class actor():
    def __init__(self,
                 act_dim,
                 obs_dim,
                 arch     = None,
                 lr       = None,
                 grd_clip = None,
                 hid_init = None,
                 fnl_init = None,
                 hid_act  = None,
                 fnl_act  = None,
                 loss     = None,
                 pol_type = None):

        # Handle arguments
        if (arch     is None): arch     = [32,32]
        if (lr       is None): lr       = 1.0e-3
        if (grd_clip is None): grd_clip = 0.1
        if (hid_init is None): hid_init = Orthogonal(gain=1.0)
        if (fnl_init is None): fnl_init = Orthogonal(gain=0.01)
        if (hid_act  is None): hid_act  = "tanh"
        if (fnl_act  is None): fnl_act  = "softmax"
        if (loss     is None): loss     = "ppo"
        if (pol_type is None): pol_type = "multinomial"

        # Fill structure
        self.loss    = loss
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.pol     = policy(pol_type, act_dim)

        # Define and init network
        self.net = network(obs_dim, self.pol.dim, arch, lr,
                           grd_clip, hid_init, fnl_init,
                           hid_act, fnl_act)

        # Define old network for PPO loss
        if (loss == "ppo"):
            self.pnet = network(obs_dim, self.pol.dim, arch, lr,
                                grd_clip, hid_init, fnl_init,
                                hid_act, fnl_act)
            self.save_weights()
            self.set_weights()

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
    def get_action(self, obs):

        # Cast
        obs = tf.cast([obs], tf.float32)

        # Forward pass to get policy parameters
        policy_params = self.call(obs)
        action        = self.pol.call(policy_params)

        return action

    # Save actor weights
    def save_weights(self):

        self.weights = self.net.get_weights()

    # Set actor weights
    def set_weights(self):

        self.pnet.set_weights(self.weights)

    # Get current learning rate
    def get_lr(self):

        return self.opt._decayed_lr(tf.float32)
