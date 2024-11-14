# Custom imports
from dragonfly.src.value.base import *

###############################################
### v_value class
class v_value(base_value):
    def __init__(self, inp_dim, inp_shape, pms, target=False):

        # Fill structure
        self.inp_dim   = inp_dim
        self.inp_shape = inp_shape
        self.out_dim   = 1
        self.target    = target

        self.net = net_factory.create(pms.network.type,
                                      inp_dim   = self.inp_dim,
                                      inp_shape = self.inp_shape,
                                      out_dim   = [self.out_dim],
                                      pms       = pms.network)

        if (self.target):
            self.tgt = net_factory.create(pms.network.type,
                                          inp_dim   = self.inp_dim,
                                          inp_shape = self.inp_shape,
                                          out_dim   = [self.out_dim],
                                          pms       = pms.network)
            self.copy_tgt()

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.net.trainables())

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)
