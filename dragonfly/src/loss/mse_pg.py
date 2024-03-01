# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for policy gradient-style value networks
class mse_pg():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, tgt, net, opt):
        with tf.GradientTape() as tape:

            # Compute loss
            val  = tf.reshape(net.call(obs), [tf.size(tgt)])
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            var   = net.trainables()
            grads = tape.gradient(loss, var)
        opt.apply_grads(zip(grads, var))
