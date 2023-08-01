# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Beta policy class (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class beta(base_policy):
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.store_dim  = self.act_dim
        self.store_type = float

        # Define and init network
        if (pms.network.heads.final[0] != "softplus"):
            warning("beta", "__init__",
                    "Final activations for beta policy network is not softplus")
        if (pms.network.heads.final[1] != "softplus"):
            warning("beta", "__init__",
                    "Final activations for beta policy network is not softplus")

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
        act      = tf.scalar_mul(2.0,act)
        act      = tf.add(act,-1.0)
        act      = np.reshape(act.numpy(), (-1,self.store_dim))
        lgp      = np.reshape(lgp.numpy(), (-1))

        return act, lgp

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Generate pdf
        self.compute_pdf([obs])

        # Sample actions
        act = self.pdf.sample(1)
        act = tf.reshape(act, [-1,self.store_dim])
        lgp = self.log_prob(act)
        lgp = tf.reshape(lgp, [-1,1])

        return act, lgp

    # Compute pdf
    def compute_pdf(self, obs):

        # Get pdf
        alpha, beta = self.forward(obs)
        alpha       = 1.0+tf.scalar_mul(2.0,alpha)
        beta        = 1.0+tf.scalar_mul(2.0,beta)
        self.pdf    = tfd.Beta(concentration1 = alpha,
                               concentration0 = beta)

        #return pdf

    # Networks forward pass
    @tf.function
    def forward(self, state):

        out   = self.net.call(state)
        alpha = out[0]
        beta  = out[1]

        return alpha, beta

    # Reshape actions
    def reshape_actions(self, act):

        return tf.reshape(act, [-1, self.act_dim])
