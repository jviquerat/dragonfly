# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Beta policy class (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class beta(base_policy):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.store_dim  = self.act_dim
        self.store_type = float
        self.pdf        = None
        self.target     = target

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

        if (self.target):
            self.tgt = net_factory.create(pms.network.type,
                                          inp_dim = obs_dim,
                                          out_dim = [self.dim,self.dim],
                                          pms     = pms.network)
            self.copy_tgt()

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

        act, lgp = self.sample(obs)
        act      = tf.scalar_mul(2.0,act)
        act      = tf.add(act,-1.0)
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

    # # Map actiona from [-1,1] to natural range
    # def map_act(self, act):

    #     a = tf.add(1.0,act)
    #     a = tf.scalar_mul(0.5,a)

    #     return a

    # Compute pdf
    def compute_pdf(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Get pdf
        alpha, beta = self.forward(obs)
        alpha       = 1.0+tf.scalar_mul(2.0,alpha)
        beta        = 1.0+tf.scalar_mul(2.0,beta)
        pdf         = tfd.Beta(concentration1 = alpha,
                               concentration0 = beta)

        return pdf

    # Networks forward pass
    def forward(self, state):

        out   = self.net.call(state)
        alpha = out[0]
        beta  = out[1]

        return alpha, beta
