# Generic imports
import numpy as np

# Custom imports
from dragonfly.core.network   import *
from dragonfly.core.optimizer import *

###############################################
### Critic class
### obs_dim : input dimension
### pms     : parameters
class critic():
    def __init__(self, obs_dim, pms):

        # Handle arguments
        loss     = "mse"

        # Fill structure
        self.dim     = 1
        self.loss    = loss
        self.obs_dim = obs_dim

        # Define network
        self.net  = network(obs_dim, self.dim, pms.network)

        # Define optimizer
        self.opt = optimizer(pms.optimizer,
                             self.net.trainable_weights)

    # Network forward pass
    def call(self, state):

        return self.net.call(state)

    # Get value from network
    def get_value(self, state):

        # Reshape state
        state = tf.cast(state, tf.float32)

        # Predict value
        val   = np.array(self.call(state))

        return val

    # Get current learning rate
    def get_lr(self):

        return self.opt.get_lr()

    # MSE loss function for critic
    @tf.function
    def train(self, obs, tgt, btc):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.convert_to_tensor(self.call(obs))
            val  = tf.reshape(val, [btc])
            p1   = tf.square(tgt - val)
            loss = tf.reduce_mean(p1)

            # Apply gradients
            crt_var     = self.net.trainable_variables
            grads       = tape.gradient(loss, crt_var)
            norm        = tf.linalg.global_norm(grads)
        self.opt.apply_grads(zip(grads,crt_var))

        return loss, norm

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()
