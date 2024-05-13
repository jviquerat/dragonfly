# Tensorflow imports
import tensorflow as tf

###############################################
### Q-policy loss class for SAC policy update
class q_pol_sac():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, p, q1, q2, alpha, opt):
        with tf.GradientTape() as tape:

            # Compute loss
            act, lgp = p.sample(obs)
            cct      = tf.concat([obs,act], axis=-1)
            tgt1     = tf.reshape(q1.call(cct), [-1,1])
            tgt2     = tf.reshape(q2.call(cct), [-1,1])
            tgt      = tf.minimum(tgt1, tgt2)
            loss     =-tf.reduce_mean(tgt - alpha*lgp)

            # Apply gradients
            var   = p.net.trainables()
            grads = tape.gradient(loss, var)

        opt.apply_grads(zip(grads, var))
