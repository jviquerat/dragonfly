# Custom imports
from dragonfly.src.policy.base import *

###############################################
### Normal policy class with full cov. matrix (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal_cma(base_policy):
    def __init__(self, obs_dim, act_dim, pms):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.dim         = self.act_dim
        self.cov_dim     = math.floor(self.dim*(self.dim - 1)/2)
        self.store_dim   = self.act_dim
        self.store_type  = float

        # Check parameters
        # if (pms.network.heads.final[0] != "tanh"):
        #     warning("normal", "__init__",
        #             "Final activation for mean of policy is not tanh")

        # if (pms.network.heads.final[1] != "sigmoid"):
        #     warning("normal", "__init__",
        #             "Final activation for stddev of policy is not sigmoid")

        # Define and init network
        self.mu_net = net_factory.create(pms.mu_network.type,
                                         inp_dim = obs_dim,
                                         out_dim = [self.dim],
                                         pms     = pms.mu_network)

        # Define and init network
        self.cov_net = net_factory.create(pms.cov_network.type,
                                          inp_dim = obs_dim,
                                          out_dim = [self.dim, self.cov_dim],
                                          pms     = pms.cov_network)

        # Define trainables
        self.mu_trainables  = self.mu_net.trainable_weights
        self.cov_trainables = self.cov_net.trainable_weights

        # Define optimizers
        self.mu_opt = opt_factory.create(pms.optimizer.type,
                                         pms       = pms.optimizer,
                                         grad_vars = self.mu_trainables)
        self.cov_opt = opt_factory.create(pms.optimizer.type,
                                          pms       = pms.optimizer,
                                          grad_vars = self.cov_trainables)

        # Define loss
        self.mu_loss = loss_factory.create(pms.loss.type,
                                           pms = pms.loss)
        self.cov_loss = loss_factory.create(pms.loss.type,
                                            pms = pms.loss)

    # Reset
    def reset(self):

        self.mu_net.reset()
        self.cov_net.reset()
        self.mu_opt.reset()
        self.cov_opt.reset()
        self.pdf = None

    # Save
    def save(self, filename):

        self.mu_net.save_weights(filename)
        self.cov_net.save_weights(filename)

    # Get actions
    def actions(self, obs):

        obs      = tf.cast(obs, tf.float32)
        act, lgp = self.sample(obs)
        act      = np.reshape(act.numpy(), (-1,self.store_dim))
        lgp      = np.reshape(lgp.numpy(), (-1))

        return act, lgp

    # Control (deterministic actions)
    def control(self, obs):

        obs        = tf.cast(obs, tf.float32)
        mu, sg, cr = self.forward(obs)
        act        = np.reshape(mu.numpy(), (-1,self.store_dim))

        return act

    # Sample actions
    @tf.function
    def sample(self, obs):

        # Generate pdf
        pdf = self.compute_pdf(obs)

        # Sample actions
        act = pdf.sample(1)
        act = tf.reshape(act, [-1,self.store_dim])
        lgp = pdf.log_prob(act)
        lgp = tf.reshape(lgp, [-1,1])

        return act, lgp

    # Compute pdf
    def compute_pdf(self, obs):

        # Get pdf
        mu     = self.forward_mu(obs)
        sg, cr = self.forward_cov(obs)
        cov    = self.get_cov(sg[0], cr[0])
        scl    = tf.linalg.cholesky(cov)
        pdf    = tfd.MultivariateNormalTriL(mu, scl)

        return pdf

    # Compute covariance matrix
    def get_cov(self, sg, cr):

        # Extract sigmas and thetas
        sigmas = sg
        thetas = cr*math.pi

        # Build initial theta matrix
        t   = tf.ones([self.dim,self.dim])*math.pi/2.0
        t   = tf.linalg.set_diag(t, tf.zeros(self.dim), k=0)
        idx = 0
        for dg in range(self.dim-1):
            diag = thetas[idx:idx+self.dim-(dg+1)]
            idx += self.dim-(dg+1)
            t    = tf.linalg.set_diag(t, diag, k=-(dg+1))
        cor = tf.cos(t)

        # Correct upper part to exact zero
        for dg in range(self.dim-1):
            size = self.dim-(dg+1)
            cor  = tf.linalg.set_diag(cor, tf.zeros(size), k=(dg+1))

        # Roll and compute additional terms
        for roll in range(self.dim-1):
            vec = tf.ones([self.dim, 1])
            vec = tf.scalar_mul(math.pi/2, vec)
            t   = tf.concat([vec, t[:, :self.dim-1]], axis=1)
            for dg in range(self.dim-1):
                zero = tf.zeros(self.dim-(dg+1))
                t    = tf.linalg.set_diag(t, zero, k=dg+1)
            cor = tf.multiply(cor, tf.sin(t))

        cor = tf.matmul(cor, tf.transpose(cor))
        scl = tf.zeros([self.dim, self.dim])
        scl = tf.linalg.set_diag(scl, sigmas, k=0)
        cov = tf.matmul(scl, cor)
        cov = tf.matmul(cov, scl)

        return cov

    # Networks forward pass
    @tf.function
    def forward_mu(self, state):

        out = self.mu_net.call(state)
        mu  = out[0]

        return mu

    # Networks forward pass
    @tf.function
    def forward_cov(self, state):

        out = self.cov_net.call(state)
        sg  = out[0]
        cr  = out[1]

        return sg, cr

    # Reshape actions
    def reshape_actions(self, act):

        return tf.reshape(act, [-1, self.act_dim])
