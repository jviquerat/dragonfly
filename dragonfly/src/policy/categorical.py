# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Categorical policy class (discrete)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class categorical(base_policy):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.store_dim  = 1
        self.store_type = int
        self.target     = target

        # Define and init network
        if (pms.network.heads.final[0] != "softmax"):
            warning("categorical", "__init__",
                    "Chosen final activation for categorical policy is not softmax")

        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = [self.dim],
                                      pms     = pms.network)

        if (self.target):
            self.tgt = net_factory.create(pms.network.type,
                                          inp_dim = obs_dim,
                                          out_dim = [self.dim],
                                          pms     = pms.network)
            self.copy_tgt()

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.net.trainables())

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

    # Get actions
    def actions(self, obs):

        obs      = tf.cast(obs, tf.float32)
        act, lgp = self.sample(obs)
        act      = np.reshape(act.numpy()[0], (-1))
        lgp      = np.reshape(lgp.numpy()[0], (-1))

        return act, lgp

    # Control (deterministic actions)
    def control(self, obs):

        obs   = tf.cast(obs, tf.float32)
        probs = self.forward(obs)
        act   = tf.argmax(probs[0][0])
        act   = np.reshape(act.numpy(), (-1))

        return act

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Generate pdf
        pdf = self.compute_pdf(obs)

        # Sample actions
        act = pdf.sample(1)
        lgp = pdf.log_prob(act)

        return act, lgp

    # Compute pdf
    def compute_pdf(self, obs):

        probs = self.forward(obs)
        return tfd.Categorical(probs=probs[0])

    # Network forward pass
    @tf.function
    def forward(self, obs):

        return self.net.call(obs)

    # Reshape actions
    def reshape_actions(self, act):

        return tf.reshape(act, [-1])
