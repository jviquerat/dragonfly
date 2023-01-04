# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Normal policy class (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal(base_policy):
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.store_dim  = self.act_dim
        self.store_type = float
        self.stddev_gen = "regular"

        # Optional stddev generation method
        if hasattr(pms, "stddev_gen"): self.stddev_gen = pms.stddev_gen
        self.stddev_log_clip = -20.0

        # Check parameters
        if (pms.network.heads.final[0] != "tanh"):
            warning("normal", "__init__",
                    "Final activation for mean network of normal policy is not tanh")

        if (self.stddev_gen == "regular"):
            if (pms.network.heads.final[1] != "sigmoid"):
                warning("normal", "__init__",
                        "Final activation for dev network of normal policy is not sigmoid")
        if (self.stddev_gen == "log"):
            if (pms.network.heads.final[1] != "linear"):
                warning("normal", "__init__",
                        "Final activation for dev network of normal policy is not linear")

        # Define and init network
        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = [self.dim,self.dim],
                                      pms     = pms.network)

        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizers
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainables)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

    # Get actions
    def actions(self, obs):

        obs      = tf.cast(obs, tf.float32)
        act, lgp = self.sample(obs)
        act      = np.reshape(act.numpy(), (-1,self.store_dim))
        lgp      = np.reshape(lgp.numpy(), (-1))

        return act, lgp

    # Control (deterministic actions)
    def control(self, obs):

        obs    = tf.cast(obs, tf.float32)
        mu, sg = self.forward(obs)
        act    = np.reshape(mu.numpy(), (-1,self.store_dim))

        return act

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Generate pdf
        pdf = self.compute_pdf(obs)

        # Sample actions
        act = pdf.sample(1)
        act = tf.reshape(act, [-1,self.store_dim])
        lgp = pdf.log_prob(act)
        lgp = tf.reshape(lgp, [-1,1])

        return act, lgp

    # Compute pdf
    def compute_pdf(self, obs):

        # Get pdf
        mu, sg = self.forward(obs)
        pdf    = tfd.MultivariateNormalDiag(loc        = mu,
                                            scale_diag = sg)

        return pdf

    # Networks forward pass
    def forward(self, state):

        out    = self.net.call(state)
        mu     = out[0]

        if (self.stddev_gen == "regular"):
            sg = out[1]
        if (self.stddev_gen == "log"):
            log_sg = out[1]
            log_sg = tf.clip_by_value(log_sg, self.stddev_log_clip, 0.0)
            sg     = tf.exp(log_sg)

        return mu, sg

    # Reshape actions
    def reshape_actions(self, act):

        return tf.reshape(act, [-1, self.act_dim])
