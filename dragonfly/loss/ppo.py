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
            pdf, prp = policy.compute_pdf(obs, True)
            if (policy.kind == "discrete"):
                act = tf.reshape(act, [-1])
            if (policy.kind == "continuous"):
                act = tf.reshape(act, [-1,policy.act_dim])
            lgp      = pdf.log_prob(act)
            prv_lgp  = prp.log_prob(act)
            ratio    = tf.exp(lgp - prv_lgp)

            # Compute actor loss
            p1       = tf.multiply(adv,ratio)
            p2       = tf.clip_by_value(ratio,
                                        1.0-self.pol_clip,
                                        1.0+self.pol_clip)
            p2       = tf.multiply(adv,p2)
            loss_ppo =-tf.reduce_mean(tf.minimum(p1,p2))

            # Compute entropy loss
            entropy      = pdf.entropy()
            entropy      = tf.reshape(entropy, [-1])
            entropy      = tf.reduce_mean(entropy, axis=0)
            loss_entropy =-entropy

            # Compute total loss
            loss = loss_ppo + self.ent_coef*loss_entropy

            # Compute KL div
            kl = pdf.kl_divergence(prp)
            kl = tf.reduce_mean(kl)

            # Apply gradients
            pol_var = policy.net.trainable_variables
            grads   = tape.gradient(loss, pol_var)
            norm    = tf.linalg.global_norm(grads)
        policy.opt.apply_grads(zip(grads, pol_var))

        return loss, kl, norm, entropy
