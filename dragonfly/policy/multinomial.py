# Generic imports
import numpy                as np
from   copy import deepcopy as cp

# Custom imports
from dragonfly.policy.tfd          import *
from dragonfly.network.network     import *
from dragonfly.optimizer.optimizer import *
from dragonfly.loss.loss           import *

###############################################
### Multinomial policy class (discrete)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class multinomial():
    def __init__(self, obs_dim, act_dim, pms):

        # Set default values
        self.save = False

        # Check inputs
        if hasattr(pms, "save"): self.save = pms.save

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.store_dim  = 1
        self.store_type = int
        self.pdf        = None

        # Define and init network
        # Force softmax activation, as this is multinomial policy
        pms.network.fnl_actv = "softmax"
        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = self.dim,
                                      pms     = pms.network)

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.net.trainable_weights)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

        # Optional previous version of network
        if (self.save): self.prv = cp(self.net)

    # Get actions
    def get_actions(self, obs):

        # Cast
        obs = tf.cast([obs], tf.float32)

        # Forward pass to get policy parameters
        policy_params = self.call(obs)

        # Sanitize output
        policy       = tf.cast(policy_params, tf.float64)
        policy, norm = tf.linalg.normalize(policy, ord=1)

        # Get actions
        self.pdf = tfd.Multinomial(1, probs=policy)
        actions  = self.pdf.sample(1)

        return actions

    # Call loss for training
    def train(self, obs, adv, act):

        return self.loss.train(obs, adv, act, self)

    # Network forward pass
    def call(self, state):

        return self.net.call(state)

    # Previous network forward pass
    def call_prv(self, state):

        return self.prv.call(state)

    # Save network weights
    def save_weights(self):

        self.weights = self.net.get_weights()

    # Set previous network weights
    def set_prv_weights(self):

        self.prv.set_weights(self.weights)

    # Get current learning rate
    def get_lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()
        if (self.save): self.prv.reset()
