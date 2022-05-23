# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Categorical policy class (discrete)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class categorical(base_policy):
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.store_dim  = 1
        self.store_type = int
        self.pdf        = None
        self.kind       = "discrete"

        # Define and init network
        if (pms.network.heads.final[0] != "softmax"):
            warning("categorical", "__init__",
                    "Chosen final activation for categorical policy is not softmax")

        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = [self.dim],
                                      pms     = pms.network)

        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.net.trainable_weights)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

    # Get actions
    def get_actions(self, obs):

        act, lgp = self.sample(obs)
        act      = np.reshape(act.numpy(), (self.store_dim))

        return act, lgp

    # Control (deterministic actions)
    def control(self, obs):

        obs   = tf.cast([obs], tf.float32)
        probs = self.call_net(obs)
        act   = tf.argmax(probs[0][0])
        act   = np.reshape(act.numpy(), (self.store_dim))

        return act

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
        probs = self.call_net(obs)
        pdf   = tfd.Categorical(probs=probs)

        return pdf

    # Network forward pass
    def call_net(self, obs):

        return self.net.call(obs)

    # Compute policy entropy
    def entropy(self, obs):

        pdf = self.compute_pdf([obs])
        return tf.get_static_value(pdf.entropy())[0][0]

    # Call loss for training
    def train(self, obs, adv, act, lgp):

        act = tf.reshape(act, [-1])
        adv = tf.reshape(adv, [-1])
        lgp = tf.reshape(lgp, [-1])
        return self.loss.train(obs, adv, act, lgp, self)
