# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.network.network     import *
from dragonfly.src.optimizer.optimizer import *
from dragonfly.src.loss.loss           import *

###############################################
### v_value class
### obs_dim : input  dimension
### pms     : parameters
class v_value():
    def __init__(self, obs_dim, pms):

        # Fill structure
        self.dim = 1
        self.obs_dim = obs_dim

        # Define and init network
        # Force linear activation, as this is v-value network
        pms.network.fnl_actv = "linear"
        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = [self.dim],
                                      pms     = pms.network)

        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainables)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

    # Get values
    def get_values(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Predict values
        values = np.array(self.call_net(obs))
        values = np.reshape(values, (-1,1))

        return values

    # Call loss for training
    def train(self, obs, tgt, size):

        return self.loss.train(obs, tgt, size, self)

    # Network forward pass
    def call_net(self, state):

        return self.net.call(state)

    # Save network weights
    def save_weights(self):

        self.weights = self.net.get_weights()

    # Set network weights
    def set_weights(self, weights):

        self.net.set_weights(weights)

    # Get current learning rate
    def get_lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()
