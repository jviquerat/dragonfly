# Generic imports
import numpy                as np
from   copy import deepcopy as cp

# Custom imports
from dragonfly.policy.tfd          import *
from dragonfly.network.network     import *
from dragonfly.optimizer.optimizer import *
from dragonfly.loss.loss           import *

###############################################
### Categorical policy class (discrete)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class categorical():
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
        self.kind       = "discrete"

        # Define and init network
        # Force softmax activation, as this is multinomial policy
        pms.network.fnl_actv = "softmax"
        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = [self.dim],
                                      pms     = pms.network)

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.net.trainable_weights)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

        # Optional previous version of network and pdf
        if (self.save):
            self.prn = cp(self.net)
            self.prp = None

    # Get actions
    def get_actions(self, obs):

        # Generate pdf
        self.pdf = self.compute_pdf([obs], False)

        # Sample actions
        actions = self.pdf.sample(1)
        actions = actions.numpy()
        actions = np.reshape(actions, (self.store_dim))

        return actions

    # Compute pdf
    def compute_pdf(self, obs, previous=False):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Get pdf
        probs = self.call_net(obs)
        pdf   = tfd.Categorical(probs=probs, validate_args=True)

        # If previous pdf is needed
        if previous:
            probs = self.call_prn(obs)
            prp   = tfd.Categorical(probs=probs)

            return pdf, prp
        else:
            return pdf

    # Call loss for training
    def train(self, obs, adv, act):

        return self.loss.train(obs, adv, act, self)

    # Network forward pass
    def call_net(self, state):

        return self.net.call(state)

    # Previous network forward pass
    def call_prn(self, state):

        return self.prn.call(state)

    # Save previous policy
    def save_prv(self):

        self.weights  = self.net.get_weights()
        self.save_pdf = cp(self.pdf)

        # On first call, prp is not initialized yet
        if (self.prp) is None:
            self.prp = cp(self.pdf)

    # Set previous policy
    def set_prv(self):

        self.prn.set_weights(self.weights)
        self.prp = cp(self.save_pdf)

    # Get current learning rate
    def get_lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()
        self.pdf = None

        if (self.save):
            self.prn.reset()
            self.prp = None
