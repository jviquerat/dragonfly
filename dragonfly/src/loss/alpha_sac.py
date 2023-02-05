# Tensorflow imports
import tensorflow as tf

import numpy as np

###############################################
### alpha loss for SAC
class alpha_sac():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, lgp, log_alpha, target, opt):
        with tf.GradientTape() as tape:

            # Compute loss
            alpha = tf.exp(log_alpha[0])
            tgt  =-alpha*tf.stop_gradient(lgp + target)
            loss = tf.reduce_mean(tgt)

            # Apply gradients
            grads = tape.gradient(loss, log_alpha)
        opt.apply_grads(zip(grads, log_alpha))
