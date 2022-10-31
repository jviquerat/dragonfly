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
    def train(self, lgp, alpha, target, opt):
        with tf.GradientTape() as tape:

            # Compute loss
            tgt  =-alpha[0]*(lgp + target)
            loss =-tf.reduce_mean(tgt)

            # Apply gradients
            grads   = tape.gradient(loss, alpha)
        opt.apply_grads(zip(grads, alpha))
