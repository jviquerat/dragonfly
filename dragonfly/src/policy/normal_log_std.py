# Custom imports
from dragonfly.src.policy.base   import *
from dragonfly.src.policy.normal import *

###############################################
### Normal policy class (continuous) with log-based std
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal_log_std(normal):
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.dim         = self.act_dim
        self.store_dim   = self.act_dim
        self.store_type  = float

        # Bounds for log std
        self.std_log_max = 2.0
        self.std_log_min =-5.0
        if hasattr(pms, "std_log_max"): self.std_log_max = pms.std_log_max
        if hasattr(pms, "std_log_min"): self.std_log_min = pms.std_log_min

        # Check parameters
        if (pms.network.heads.final[0] != "linear"):
            warning("normal_log_std", "__init__",
                    "Final activation for mean of policy is not linear")

        if (pms.network.heads.final[1] != "sigmoid"):
            warning("normal_log_std", "__init__",
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

    # Networks forward pass
    @tf.function
    def forward(self, state):

        out    = self.net.call(state)
        mu     = out[0]
        log_sg = out[1]
        log_sg = self.std_log_min + log_sg*(self.std_log_max - self.std_log_min)
        sg     = tf.exp(log_sg)

        return mu, sg
