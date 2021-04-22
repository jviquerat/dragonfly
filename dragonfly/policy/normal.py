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

        # Define and init mu network
        # Force tanh activation, as this is normal policy
        pms.mu_network.fnl_actv = "tanh"
        self.mu_net = net_factory.create(pms.mu_network.type,
                                         inp_dim = obs_dim,
                                         out_dim = self.dim,
                                         pms     = pms.mu_network)

        # Define and init sg network
        # Force sigmoid activation, as this is normal policy
        pms.sg_network.fnl_actv = "softplus"
        self.sg_net = net_factory.create(pms.sg_network.type,
                                         inp_dim = obs_dim,
                                         out_dim = self.dim,
                                         pms     = pms.sg_network)

        # Define trainable variables
        self.trainable = tf.concat([self.mu_net.trainable_variables,
                                    self.sg_net.trainable_variables], 0)

        print(self.trainable)

        # Define optimizers
        #self.mu_opt = opt_factory.create(pms.optimizer.type,
        #                                 pms       = pms.optimizer,
        #                                 grad_vars = self.mu_net.trainable_weights)

        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainable)

        #self.sg_opt = opt_factory.create(pms.optimizer.type,
        #                                 pms       = pms.optimizer,
        #                                 grad_vars = self.sg_net.trainable_weights)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

        # Optional previous version of networks
        if (self.save):
            self.mu_prn = cp(self.mu_net)
            self.sg_prn = cp(self.sg_net)
            self.prp = None

    # Get actions
    def get_actions(self, obs):

        # Generate pdf
        self.pdf = self.compute_pdf(obs, False)

        # Sample actions
        actions = self.pdf.sample(self.act_dim)
        #actions = actions.numpy()
        #actions = np.reshape(actions, (self.store_dim))

        return actions

    # Compute pdf
    def compute_pdf(self, obs, previous=False):

        # Cast
        obs = tf.cast([obs], tf.float32)

        # Get pdf
        mu, sg = self.call_nets(obs)
        pdf    = tfd.MultivariateNormalDiag(loc        = mu,
                                            scale_diag = sg)

        # If previous pdf is needed
        if previous:
            pmu, psg = self.call_prns(obs)
            prp      = tfd.MultivariateNormalDiag(loc        = mu,
                                                  scale_diag = sg)

            return pdf, prp
        else:
            return pdf

    # Call loss for training
    def train(self, obs, adv, act):

        return self.loss.train(obs, adv, act, self)

    # Networks forward pass
    def call_nets(self, state):

        mu  = self.mu_net.call(state)
        sg  = self.sg_net.call(state)

        return mu, sg

    # Previous networks forward pass
    def call_prns(self, state):

        pmu = self.mu_prn.call(state)
        psg = self.sg_prn.call(state)

        return pmu, psg

    # Save previous policy
    def save_prv(self):

        self.mu_weights = self.mu_net.get_weights()
        self.sg_weights = self.sg_net.get_weights()
        self.save_pdf   = cp(self.pdf)

        # On first call, prp is not initialized yet
        if (self.prp) is None:
            self.prp = cp(self.pdf)

    # Set previous policy
    def set_prv(self):

        self.mu_prn.set_weights(self.mu_weights)
        self.sg_prn.set_weights(self.sg_weights)
        self.prp = cp(self.save_pdf)

    # Get current learning rate
    def get_lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.mu_net.reset()
        self.sg_net.reset()
        self.mu_opt.reset()
        self.sg_opt.reset()
        self.pdf = None

        if (self.save):
            self.mu_prn.reset()
            self.sg_prn.reset()
            self.prp = None

