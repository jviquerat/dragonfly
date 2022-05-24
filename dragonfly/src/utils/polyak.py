# Generic imports
import numpy as np

###############################################
### Polyak averager for neural networks
class polyak:
    def __init__(self, rho):

        self.rho = rho

    # Update network by polyak average
    def average(self, net, tgt):

        v = net.get_weights()
        t = tgt.get_weights()
        w = [self.rho*wvv + (1.0-self.rho)*wtt for wvv, wtt in zip(v, t)]

        tgt.set_weights(w)

