# Tensorflow imports
import tensorflow as tf

###############################################
### MSE loss class for TD3-style q networks
class mse_td3():
    def __init__(self, pms):
        pass

    # Train
    @tf.function
    def train(self,
              obs, nxt, act, rwd, trm, gamma, sigma, clp,
              p_tgt, q, q1_tgt, q2_tgt, opt):
        with tf.GradientTape() as tape:

            # Compute target
            nac  = tf.reshape(p_tgt.call(nxt), [tf.size(rwd),-1])
            nse  = tf.random.normal(tf.shape(nac), 0.0, sigma)
            nse  = tf.clip_by_value(nse, -clp, clp)
            nac  = tf.clip_by_value(nac+nse, -1.0, 1.0)
            nct  = tf.concat([nxt, nac], axis=-1)
            tgt1 = tf.reshape(q1_tgt.call(nct), [-1,1])
            tgt2 = tf.reshape(q2_tgt.call(nct), [-1,1])
            tgt  = tf.minimum(tgt1, tgt2)
            trm  = tf.clip_by_value(trm, 0.0, 1.0)
            tgt  = rwd + trm*gamma*tgt

            # Compute loss
            oac  = tf.concat([obs, act], axis=-1)
            val  = tf.reshape(q.call(oac), [-1,1])
            diff = tf.square(tgt - val)
            loss = tf.reduce_mean(diff)

            # Apply gradients
            var   = q.trainables()
            grads = tape.gradient(loss, var)
        opt.apply_grads(zip(grads, var))
