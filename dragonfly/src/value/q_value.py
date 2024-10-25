# Custom imports
from dragonfly.src.value.base import *

###############################################
### q_value class
### inp_dim : input  dimension
### out_dim : output dimension
### pms     : parameters
class q_value(base_value):
    def __init__(self, inp_dim, inp_shape, out_dim, pms, target=False):

        # Fill structure
        self.inp_dim   = inp_dim
        self.inp_shape = inp_shape
        self.out_dim   = out_dim
        self.target    = target

        # Define and init network
        if (pms.network.heads.final[0] != "linear"):
            warning("q_value", "__init__",
                    "Chosen final activation for q_value is not linear")

        self.net = net_factory.create(pms.network.type,
                                      inp_dim   = self.inp_dim,
                                      inp_shape = self.inp_shape,
                                      out_dim   = [self.out_dim],
                                      pms       = pms.network)

        if (self.target):
            self.tgt = net_factory.create(pms.network.type,
                                          inp_dim = self.inp_dim,
                                          inp_shape = self.inp_shape,
                                          out_dim = [self.out_dim],
                                          pms     = pms.network)
            self.copy_tgt()

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.net.trainables())

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)
