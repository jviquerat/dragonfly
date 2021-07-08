# Tensorflow imports
import tensorflow as tf

# Custom imports
from dragonfly.core.constants import *

###############################################
### PG loss class
class pg():
    def __init__(self, pms):

        # Set default values
        self.ent_coef   = 0.01

        # Check inputs
        if hasattr(pms, "ent_coef"): self.ent_coef = pms.ent_coef

    # Train
    @tf.function
    def train(self, obs, adv, act, policy):
        with tf.GradientTape() as tape:

            # Compute loss
            pdf     = policy.compute_pdf(obs)
            act     = policy.reshape_actions(act)
            lgp     = pdf.log_prob(act)
            loss_pg = -tf.reduce_mean(lgp)

            # Compute entropy loss
            entropy      = pdf.entropy()
            entropy      = tf.reshape(entropy, [-1])
            entropy      = tf.reduce_mean(entropy, axis=0)
            loss_entropy =-entropy

            # Compute total loss
            loss = loss_pg + self.ent_coef*loss_entropy

            # Compute KL div
            kl = pdf.kl_divergence(prp)
            kl = tf.reduce_mean(kl)

            # Apply gradients
            pol_var = policy.trainables
            grads   = tape.gradient(loss, pol_var)
            norm    = tf.linalg.global_norm(grads)
        policy.opt.apply_grads(zip(grads, pol_var))

        return loss, kl, norm, entropy
