# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class
class mse():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, tgt, btc, value):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.convert_to_tensor(value.call(obs))
            val  = tf.reshape(val, [btc])
            p1   = tf.square(tgt - val)
            loss = tf.reduce_mean(p1)

            # Apply gradients
            val_var     = value.net.trainable_variables
            grads       = tape.gradient(loss, val_var)
            norm        = tf.linalg.global_norm(grads)
        value.opt.apply_grads(zip(grads, val_var))

        return loss, norm
