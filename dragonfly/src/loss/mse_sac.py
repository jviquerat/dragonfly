# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for SAC-style q networks
class mse_sac():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self, obs, nxt, act, rwd, trm, gamma, alpha, p, q, qt1, qt2):
        with tf.GradientTape() as tape:

            # Compute target
            nac, lgp = p.sample(nxt)
            nct  = tf.concat([nxt, nac], axis=-1)
            tgt1 = qt1.forward(nct)
            tgt2 = qt2.forward(nct)
            tgt  = tf.minimum(tgt1, tgt2)
            tgt  = tgt - alpha*lgp

            tgt  = tf.reshape(tgt, [-1,1])
            trm  = tf.clip_by_value(trm, 0.0, 1.0)
            tgt  = rwd + trm*gamma*tgt

            # Compute loss
            oac  = tf.concat([obs, act], axis=-1)
            val  = tf.cast(q.forward(oac), tf.float32)
            val  = tf.reshape(val, [-1,1])
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            val_var = q.trainables
            grads   = tape.gradient(loss, val_var)
        q.opt.apply_grads(zip(grads, val_var))
