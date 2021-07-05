# Generic imports
import numpy                as np
from   copy import deepcopy as cp

# Custom imports
from dragonfly.policy.tfd          import *
from dragonfly.network.network     import *
from dragonfly.optimizer.optimizer import *
from dragonfly.loss.loss           import *

###############################################
### Normal policy class (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal():
    def __init__(self, obs_dim, act_dim, pms):

        # Set default values
        self.save = False

        # Check inputs
        if hasattr(pms, "save"): self.save = pms.save

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.store_dim  = self.act_dim
        self.store_type = float
        self.pdf        = None
        self.kind       = "continuous"

        # Define and init network
        # Force tanh activation, as this is normal policy
        #pms.network.heads.final = ["tanh","sigmoid"]
        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = [self.dim,self.dim],
                                      pms     = pms.network)

        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizers
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainables)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

        # Optional previous version of networks
        if (self.save):
            self.prn = cp(self.net)
            self.prp = None

    # Get actions
    def get_actions(self, obs):

        # Generate pdf
        self.pdf = self.compute_pdf([obs])

        # Sample actions
        actions = self.pdf.sample(self.act_dim)
        actions = tf.clip_by_value(actions, -1.0, 1.0)
        actions = actions.numpy()
        actions = np.reshape(actions, (self.store_dim))

        return actions

    # Compute pdf
    def compute_pdf(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Get pdf
        mu, sg = self.call_net(obs)
        pdf    = tfd.MultivariateNormalDiag(loc        = mu,
                                            scale_diag = sg)

        return pdf

    # Compute previous pdf
    def compute_prp(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Get pdf
        mu, sg = self.call_prn(obs)
        pdf    = tfd.MultivariateNormalDiag(loc        = mu,
                                            scale_diag = sg)

        return pdf

    # Reshape actions for training
    def reshape_actions(self, act):

        return tf.reshape(act, [-1, self.act_dim])

    # Call loss for training
    def train(self, obs, adv, act):

        return self.loss.train(obs, adv, act, self)

    # Networks forward pass
    def call_net(self, state):

        out = self.net.call(state)
        mu  = out[0]
        sg  = tf.square(out[1])

        return mu, sg

    # Previous networks forward pass
    def call_prn(self, state):

        out = self.prn.call(state)
        mu  = out[0]
        sg  = tf.square(out[1])

        return mu, sg

    # Save previous policy
    def save_prv(self):

        self.prn.set_weights(self.net.get_weights())
        self.prp = self.pdf.copy()

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

