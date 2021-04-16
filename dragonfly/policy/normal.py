# Generic imports
import numpy as np

# Custom imports
from dragonfly.network.network     import *
from dragonfly.optimizer.optimizer import *

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
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.dim     = self.act_dim

        # Define and init mu network
        # Force tanh activation, as this is normal policy
        pms.mu_network.fnl_actv = "tanh"
        self.net_mu = net_factory.create(pms.mu_network.type,
                                         inp_dim = obs_dim,
                                         out_dim = self.dim,
                                         pms     = pms.mu_network)

        # Define and init sg network
        # Force sigmoid activation, as this is normal policy
        pms.sg_network.fnl_actv = "sigmoid"
        self.net_sg = net_factory.create(pms.sg_network.type,
                                         inp_dim = obs_dim,
                                         out_dim = self.dim,
                                         pms     = pms.sg_network)

        # Define optimizers
        self.mu_opt = opt_factory.create(pms.optimizer.type,
                                         pms       = pms.optimizer,
                                         grad_vars = self.mu_net.trainable_weights)

        self.sg_opt = opt_factory.create(pms.optimizer.type,
                                         pms       = pms.optimizer,
                                         grad_vars = self.sg_net.trainable_weights)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

        # Optional previous version of network
        if (self.save):
            self.prv = cp(self.net)

    # Call policy
    def call(self, params):

        print("Normal law policy not implemented yet")
        exit()
