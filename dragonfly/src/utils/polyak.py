# Generic imports
import numpy as np

###############################################
### Polyak averager for neural networks
class polyak:
    def __init__(self, rho):

        self.rho = rho

    # Update network by polyak average
    @tf.function
    def average(self, net, tgt):

        for wv, wt in zip(net.weights, tgt.weights):
            w = self.rho*wt + (1.0-self.rho)*wv
            wt.assign(w)

