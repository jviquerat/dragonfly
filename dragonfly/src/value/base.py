# Generic imports
import numpy      as np
import tensorflow as tf

# Custom imports
from dragonfly.src.network.network     import net_factory
from dragonfly.src.optimizer.optimizer import opt_factory
from dragonfly.src.loss.loss           import loss_factory

###############################################
### Base value
class base_value():
    def __init__(self):
        pass

    # Get values
    def values(self, obs):
        raise NotImplementedError

    # Network forward pass
    def forward(self, x):

        return self.net.call(tf.cast(x, tf.float32))[0]

    # Get values
    def values(self, x):

        return np.reshape(self.forward(x), (-1,self.out_dim))

    # Save network weights
    def save_weights(self):

        self.weights = self.net.get_weights()

    # Set network weights
    def set_weights(self, weights):

        self.net.set_weights(weights)

    # Get current learning rate
    def lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()

        if (self.target):
            self.tgt.reset()
            self.copy_tgt()

    # Save
    def save(self, filename):

        self.net.save_weights(filename)

    # Load
    def load(self, filename):

        load_status = self.net.load_weights(filename)
        load_status.assert_consumed()

    # Copy net into tgt
    def copy_tgt(self):

        self.tgt.set_weights(self.net.get_weights())
