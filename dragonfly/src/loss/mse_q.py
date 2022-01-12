# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for Q-value
class mse_q():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, act, tgt, btc, value):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.gather(value.call_net(obs), act, axis=1, batch_dims=1)
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            val_var = value.trainables
            grads   = tape.gradient(loss, val_var)
            norm    = tf.linalg.global_norm(grads)
        value.opt.apply_grads(zip(grads, val_var))

        return loss, norm
