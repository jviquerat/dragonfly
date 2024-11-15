# Custom imports
from dragonfly.src.core.constants import *
from dragonfly.src.policy.tfd     import *
from dragonfly.src.policy.base    import base_normal

###############################################
### Tanh-normal policy class (continuous)
class tanh_normal(base_normal):
    def __init__(self, obs_dim, obs_shape, act_dim, pms, target=False):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.obs_shape   = obs_shape
        self.out_dim     = [self.act_dim, self.act_dim]
        self.store_type  = float
        self.target      = target
        self.min_log_std = pms.min_log_std
        self.max_log_std = pms.max_log_std

        # Check parameters
        if (pms.network.heads.final[0] != "linear"):
            warning("tanh_normal", "__init__",
                    "Final activation for mean of policy is not linear")

        if (pms.network.heads.final[1] != "linear"):
            warning("tanh_normal", "__init__",
                    "Final activation for stddev of policy is not linear")

        # Init from base class
        super().__init__(pms)

    # Control (deterministic actions)
    def control(self, obs):

        mu, sg   = self.forward(tf.cast(obs, tf.float32))
        act      = tf.reshape(mu, [-1,self.act_dim])
        tanh_act = tf.tanh(act)
        tanh_act = np.reshape(tanh_act.numpy(), (-1, self.act_dim))

        return tanh_act

    # Networks forward pass
    @tf.function
    def forward(self, state):

        out     = self.net.call(state)
        mu      = out[0]
        log_std = out[1]
        log_std = tf.clip_by_value(log_std,
                                   self.min_log_std,
                                   self.max_log_std)
        std     = tf.exp(log_std)

        return mu, std

    # Sample actions
    # Mostly taken from openAI implementation
    @tf.function
    def sample(self, obs):

        # Reparameterization trick
        mu, std = self.forward(obs)
        act     = mu + tf.random.normal(tf.shape(mu))*std

        # Compute gaussian likelihood
        lkh = -0.5*(((act - mu)/(std + eps))**2 +
                    2.0*tf.math.log(std) +
                    tf.math.log(2.0*math.pi))
        lgp = tf.reduce_sum(lkh, axis=1)

        # Squash actions
        tanh_act = tf.tanh(act)

        # Compute log-prob of reparameterized action
        # Regular version, possibly numerically unstable
        # sth = tf.math.log(1.0 - tf.square(tanh_act) + 1.0e-8)

        # OpenAI version, numerically stable
        sth  = 2.0*(np.log(2.0) - act - tf.nn.softplus(-2.0*act))
        sth  = tf.reduce_sum(sth, axis=1)
        sth  = tf.reshape(sth, [-1,1])
        lgp -= sth

        return tanh_act, lgp
