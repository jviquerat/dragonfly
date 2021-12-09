# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Categorical policy class (discrete)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class categorical(base_policy):
    def __init__(self, obs_dim, act_dim, pms):

        # Set default values
        self.save = False

        # Check inputs
        if hasattr(pms, "save"): self.save = pms.save

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

        # Optional previous version of network and pdf
        if (self.save):
            self.prn = cp(self.net)
            self.prp = None

    # Get actions
    def get_actions(self, obs):

        # Generate pdf
        self.pdf = self.compute_pdf([obs])

        # Sample actions
        actions = self.pdf.sample(1)
        actions = actions.numpy()
        actions = np.reshape(actions, (self.store_dim))

        return actions

    # Compute pdf
    def compute_pdf(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Get pdf
        probs = self.call_net(obs)
        pdf   = tfd.Categorical(probs=probs)

        return pdf

    # Compute previous pdf
    def compute_prp(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Get pdf
        probs = self.call_prn(obs)
        pdf   = tfd.Categorical(probs=probs)

        return pdf

    # Reshape actions for training
    def reshape_actions(self, act):

        return tf.reshape(act, [-1])

    # Network forward pass
    def call_net(self, state):

        return self.net.call(state)

    # Previous network forward pass
    def call_prn(self, state):

        return self.prn.call(state)
