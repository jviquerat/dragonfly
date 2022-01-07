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
        self.pdf        = None
        self.kind       = "continuous"

        # Define and init network
        if (pms.network.heads.final[0] != "tanh"):
            warning("normal", "__init__",
                    "Final activation for mean network of normal policy is not tanh")
        if (pms.network.heads.final[1] != "sigmoid"):
            warning("normal", "__init__",
                    "Final activation for dev network of normal policy is not sigmoid")

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
    def get_actions(self, obs):

        act, lgp = self.sample(obs)
        #act     = tf.clip_by_value(act, -1.0, 1.0)
        act      = tf.tanh(act)
        act      = np.reshape(act.numpy(), (self.store_dim))

        return act, lgp

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Generate pdf
        self.pdf = self.compute_pdf([obs])

        # Sample actions
        act = self.pdf.sample(1)
        lgp = self.pdf.log_prob(act)

        return act, lgp

    # Compute pdf
    def compute_pdf(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Get pdf
        mu, sg = self.call_net(obs)
        pdf    = tfd.MultivariateNormalDiag(loc        = mu,
                                            scale_diag = sg)

        return pdf

    # Reshape actions for training
    def reshape_actions(self, act):

        return tf.reshape(act, [-1, self.act_dim])

    # Networks forward pass
    def call_net(self, state):

        out = self.net.call(state)
        mu  = out[0]
        sg  = tf.square(out[1])

        return mu, sg
