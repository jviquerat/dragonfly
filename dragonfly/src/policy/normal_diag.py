# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Normal policy class with diagonal covariance matrix (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal_diag(base_policy):
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.dim         = self.act_dim
        self.store_dim   = self.act_dim
        self.store_type  = float
        self.sigma0      = pms.sigma0

        # Check parameters
        if (pms.network.heads.final[0] != "tanh"):
            warning("normal", "__init__",
                    "Final activation for mean of policy is not tanh")

        if (pms.network.heads.final[1] != "sigmoid"):
            warning("normal", "__init__",
                    "Final activation for stddev of policy is not sigmoid")

        # Define and init network
        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = [self.dim, self.dim],
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
        act = tf.clip_by_value(act, -1.0, 1.0)
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
    @tf.function
    def forward(self, state):

        out = self.net.call(state)
        mu  = out[0]
        sg  = out[1]*self.sigma0/0.5

        return mu, sg

    # Reshape actions
    def reshape_actions(self, act):

        return tf.reshape(act, [-1, self.act_dim])
