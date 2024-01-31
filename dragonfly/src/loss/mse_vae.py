# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for variational auto-encoder
class mse_vae():
    def __init__(self, pms):

        self.beta = pms.beta

    # Train
    @tf.function
    def train(self, x, vae):
        with tf.GradientTape() as tape:

            # Compute loss
            y, m, s = vae.forward(x)
            diff    = tf.square(y - x)
            mse     = tf.reduce_mean(diff)

            kl = 0.5*tf.reduce_sum(-1.0 - 2.0*tf.math.log(s) + s**2 + m**2, axis=1)
            kl = tf.reduce_mean(kl)

            loss = mse + self.beta*kl

            # Apply gradients
            vae_var = vae.trainables
            grads   = tape.gradient(loss, vae_var)
        vae.opt.apply_grads(zip(grads, vae_var))

        return loss
