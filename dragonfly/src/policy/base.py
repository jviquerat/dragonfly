# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.policy.tfd          import *
from dragonfly.src.network.network     import net_factory
from dragonfly.src.optimizer.optimizer import opt_factory
from dragonfly.src.loss.loss           import loss_factory

###############################################
### Base policy
class base_policy():
    def __init__(self, pms):

        self.net = net_factory.create(pms.network.type,
                                      inp_dim   = self.obs_dim,
                                      inp_shape = self.obs_shape,
                                      out_dim   = self.out_dim,
                                      pms       = pms.network)

        if (self.target):
            self.tgt = net_factory.create(pms.network.type,
                                          inp_dim   = self.obs_dim,
                                          inp_shape = self.obs_shape,
                                          out_dim   = self.out_dim,
                                          pms       = pms.network)
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
        raise NotImplementedError

    # Control (deterministic actions)
    def control(self, obs):
        raise NotImplementedError

    # Compute pdf
    def compute_pdf(self, obs):
        raise NotImplementedError

    # Reshape actions for training
    def reshape_actions(self, act):
        raise NotImplementedError

    # Networks forward pass
    def forward(self, state):
        raise NotImplementedError

    # Save network weights
    def save_weights(self):

        self.weights = self.net.get_weights()

    # Set network weights
    def set_weights(self, weights):

        self.net.set_weights(weights)

    # Get current learning rate
    def lr(self):

        return self.opt.get_lr()

    # Reset
    def reset(self):

        self.net.reset()
        self.opt.reset()
        self.pdf = None

        if (self.target):
            self.tgt.reset()
            self.copy_tgt()

    # Save
    def save(self, filename):

        self.net.save_weights(filename)

    # Load
    def load(self, filename):

        load_status = self.net.load_weights(filename)

    # Copy net into tgt
    def copy_tgt(self):

        self.tgt.set_weights(self.net.get_weights())

    # Reshape actions
    def reshape_actions(self, act):

        return tf.reshape(act, [-1, self.act_dim])

    # Random uniform actions for warmup
    def random_uniform(self, obs):

        n_cpu = obs.shape[0]
        act   = np.random.uniform(-1.0, 1.0, size=self.act_dim)
        return act

###############################################
### Base policy for normal laws
class base_normal(base_policy):
    def __init__(self, pms):

        super().__init__(pms)

    # Get actions
    def actions(self, obs):

        act, lgp = self.sample(tf.cast(obs, tf.float32))
        act      = np.reshape(act.numpy(), (-1,self.act_dim))
        lgp      = np.reshape(lgp.numpy(), (-1))

        return act, lgp

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Generate pdf
        pdf = self.compute_pdf(obs)

        # Sample actions
        act = tf.reshape(pdf.sample(1), [-1,self.act_dim])
        lgp = tf.reshape(pdf.log_prob(act), [-1,1])

        return act, lgp
