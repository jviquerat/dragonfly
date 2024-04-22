# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for auto-encoder
class mse_ae():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, x, ae):
        with tf.GradientTape() as tape:

            # Compute loss
            y    = ae.forward(x)
            diff = tf.square(y - x)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            ae_var = ae.trainables
            grads   = tape.gradient(loss, ae_var)
        ae.opt.apply_grads(zip(grads, ae_var))

        return loss
