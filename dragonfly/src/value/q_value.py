# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.network.network     import *
from dragonfly.src.optimizer.optimizer import *
from dragonfly.src.loss.loss           import *

###############################################
### q_value class
### obs_dim : input  dimension
### act_dim : action dimension
### pms     : parameters
class q_value():
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # Define and init network
        if (pms.network.heads.final[0] != "linear"):
            warning("q_value", "__init__",
                    "Chosen final activation for q_value is not linear")

        self.net = net_factory.create(pms.network.type,
                                      inp_dim = self.obs_dim,
                                      out_dim = [self.act_dim],
                                      pms     = pms.network)

        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainables)

        # Define loss
        if (pms.loss.type != "mse"):
            warning("q_value", "__init__",
                    "Chosen loss for q_value is not mse")
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

    # Get values
    def get_values(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Predict values
        values = np.array(self.call_net(obs))
        values = np.reshape(values, (-1,self.act_dim))

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
