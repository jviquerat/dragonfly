# Generic imports
import numpy as np

# Custom imports
from dragonfly.core.network     import *
from dragonfly.core.optimizer   import *
from dragonfly.policies.factory import *

###############################################
### Actor class
### act_dim : output dimension
### obs_dim : input  dimension
### pms     : parameters
class actor():
    def __init__(self, obs_dim, act_dim, pms):

        # Handle arguments
        loss     = "ppo"

        # Fill structure
        self.loss    = loss
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.pol     = policy_factory.create(pms.pol_type,
                                             act_dim = act_dim)

        # Define and init network
        self.net = network(obs_dim, self.pol.dim, pms)

        # Define old network for PPO loss
        if (loss == "ppo"):
            self.pnet = network(obs_dim, self.pol.dim, pms)
            self.save_weights()
            self.set_weights()

        # Define optimizer
        self.opt = optimizer(pms.lr, pms.grd_clip,
                             self.net.trainable_weights)

    # Network forward pass
    def call(self, state):

        # Copy inputs
        var = state

        # Call network
        var = self.net.call(var)

        return var

    # Get actions
    def get_action(self, obs):

        # Cast
        obs = tf.cast([obs], tf.float32)

        # Forward pass to get policy parameters
        policy_params = self.call(obs)
        action        = self.pol.call(policy_params)

        return action

    # Save actor weights
    def save_weights(self):

        self.weights = self.net.get_weights()

    # Set actor weights
    def set_weights(self):

        self.pnet.set_weights(self.weights)

    # Get current learning rate
    def get_lr(self):

        return self.opt.get_lr()

    # PPO loss function for actor
    @tf.function
    def train(self, obs, adv, act, pol_clip, entropy_coef):
        with tf.GradientTape() as tape:

            # Compute ratio of probabilities
            prv_pol  = tf.convert_to_tensor(self.pnet.call(obs))
            pol      = tf.convert_to_tensor(self.call(obs))
            new_prob = tf.reduce_sum(act*pol,     axis=1)
            prv_prob = tf.reduce_sum(act*prv_pol, axis=1)
            new_log  = tf.math.log(new_prob + 1.0e-5)
            old_log  = tf.math.log(prv_prob + 1.0e-5)
            ratio    = tf.exp(new_log - old_log)

            # Compute actor loss
            p1         = tf.multiply(adv,ratio)
            p2         = tf.clip_by_value(ratio,
                                          1.0-pol_clip,
                                          1.0+pol_clip)
            p2         = tf.multiply(adv,p2)
            loss_ppo   =-tf.reduce_mean(tf.minimum(p1,p2))

            # Compute entropy loss
            entropy      = tf.multiply(pol,tf.math.log(pol + 1.0e-5))
            entropy      =-tf.reduce_sum(entropy, axis=1)
            entropy      = tf.reduce_mean(entropy)
            loss_entropy =-entropy

            # Compute total loss
            loss = loss_ppo + entropy_coef*loss_entropy

            # Compute KL div
            kl = tf.math.log(pol + 1.0e-5) - tf.math.log(prv_pol + 1.0e-5)
            kl = 0.5*tf.reduce_mean(tf.square(kl))

            # Apply gradients
            act_var = self.net.trainable_variables
            grads   = tape.gradient(loss, act_var)
            norm    = tf.linalg.global_norm(grads)
        self.opt.apply_grads(zip(grads,act_var))

        return loss, kl, norm, entropy

    # Reset
    def reset(self):

        if (self.loss == "ppo"): self.pnet.reset()
        self.net.reset()
        self.opt.reset()
