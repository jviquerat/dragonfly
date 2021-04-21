# Tensorflow imports
import tensorflow as tf

# Custom imports
from dragonfly.core.constants import *

###############################################
### PPO loss class
class ppo():
    def __init__(self, pms):

        # Set default values
        self.pol_clip   = 0.2
        self.ent_coef   = 0.01

        # Check inputs
        if hasattr(pms, "pol_clip"): self.pol_clip = pms.pol_clip
        if hasattr(pms, "ent_coef"): self.ent_coef = pms.ent_coef

    # Train
    @tf.function
    def train(self, obs, adv, act, policy):
        with tf.GradientTape() as tape:

            # Compute ratio of probabilities
            prv_pol  = tf.convert_to_tensor(policy.call_prn(obs))
            pol      = tf.convert_to_tensor(policy.call(obs))
            new_prob = tf.reduce_sum(act*pol,     axis=1)
            prv_prob = tf.reduce_sum(act*prv_pol, axis=1)
            new_log  = tf.math.log(new_prob + ppo_eps)
            old_log  = tf.math.log(prv_prob + ppo_eps)
            ratio    = tf.exp(new_log - old_log)

            # Compute actor loss
            p1         = tf.multiply(adv,ratio)
            p2         = tf.clip_by_value(ratio,
                                          1.0-self.pol_clip,
                                          1.0+self.pol_clip)
            p2         = tf.multiply(adv,p2)
            loss_ppo   =-tf.reduce_mean(tf.minimum(p1,p2))

            # Compute entropy loss
            entropy      = tf.multiply(pol,tf.math.log(pol + ppo_eps))
            entropy      =-tf.reduce_sum(entropy, axis=1)
            entropy      = tf.reduce_mean(entropy)
            loss_entropy =-entropy

            # Compute total loss
            loss = loss_ppo + self.ent_coef*loss_entropy

            # Compute KL div
            kl = tf.math.log(pol + ppo_eps) - tf.math.log(prv_pol + ppo_eps)
            kl = 0.5*tf.reduce_mean(tf.square(kl))

            # Apply gradients
            pol_var = policy.net.trainable_variables
            grads   = tape.gradient(loss, pol_var)
            norm    = tf.linalg.global_norm(grads)
        policy.opt.apply_grads(zip(grads, pol_var))

        return loss, kl, norm, entropy
