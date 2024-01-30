# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for variational auto-encoder
class mse_vae():
    def __init__(self, pms):

        self.beta = pms.beta

    # Train
    @tf.function
    def train(self, x, ae):
        with tf.GradientTape() as tape:

            # Compute loss
            y, m, s = ae.forward(x)
            diff    = tf.square(y - x)
            mse     = tf.reduce_mean(diff)

            kl = 0.5*tf.reduce_sum(-1.0 - 2.0*tf.math.log(s) + s**2 + m**2, axis=1)
            kl = tf.reduce_mean(kl)

            loss = mse + self.beta*kl

            # Apply gradients
            ae_var = ae.trainables
            grads   = tape.gradient(loss, ae_var)
        ae.opt.apply_grads(zip(grads, ae_var))

        return loss
