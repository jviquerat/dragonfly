# Custom imports
from dragonfly.src.policy.tfd  import *
from dragonfly.src.policy.base import base_normal

###############################################
### Normal policy class with full covariance matrix (continuous)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class normal_full(base_normal):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim     = act_dim
        self.obs_dim     = obs_dim
        self.dim         = self.act_dim
        self.cov_dim     = math.floor(self.dim*(self.dim - 1)/2)
        self.out_dim     = [self.dim, self.dim, self.cov_dim]
        self.store_dim   = self.act_dim
        self.store_type  = float
        self.target      = target

        self.sigma       = 1.0
        if (hasattr(pms, "sigma")): self.sigma = pms.sigma

        # Check parameters
        if (pms.network.heads.final[0] != "tanh"):
            warning("normal", "__init__",
                    "Final activation for mean of policy is not tanh")

        if (pms.network.heads.final[1] != "sigmoid"):
            warning("normal", "__init__",
                    "Final activation for stddev of policy is not sigmoid")

        if (pms.network.heads.final[2] != "sigmoid"):
            warning("normal", "__init__",
                    "Final activation for correlations of policy is not sigmoid")

        # Init from base class
        super().__init__(pms)

    # Control (deterministic actions)
    def control(self, obs):

        mu, sg, cr = self.forward(tf.cast(obs, tf.float32))
        act        = np.reshape(mu.numpy(), (-1,self.store_dim))

        return act

    # Compute pdf
    def compute_pdf(self, obs):

        # Get pdf
        mu, sg, cr = self.forward(obs)
        cov        = self.get_cov(sg[0], cr[0])

        scl        = tf.linalg.cholesky(cov)
        pdf        = tfd.MultivariateNormalTriL(loc        = mu,
                                                scale_tril = scl)

        return pdf

    # Compute covariance matrix
    def get_cov(self, sg, cr):

        # # Create skew-symmetric matrix
        # t = tf.zeros([self.dim, self.dim])

        # idx = 0
        # for dg in range(self.dim-1):
        #     diag = cr[idx:idx+self.dim-(dg+1)]
        #     idx += self.dim-(dg+1)
        #     t    = tf.linalg.set_diag(t,  diag, k=-(dg+1))
        #     t    = tf.linalg.set_diag(t, -diag, k= (dg+1))

        # # Exponentiate to get orthogonal matrix
        # et = tf.linalg.expm(t)

        # # Generate diagonal matrix
        # s = tf.zeros([self.dim, self.dim])
        # s = tf.linalg.set_diag(s, sg, k=0)

        # # Generate covariance matrix
        # cov = tf.matmul(et,s)
        # cov = tf.matmul(cov, tf.transpose(et))

        # Extract sigmas and thetas
        sigmas = sg
        thetas = cr*math.pi

        #print(sg, cr)

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
        scl = tf.linalg.set_diag(scl, tf.sqrt(sigmas), k=0)
        cov = tf.matmul(scl, cor)
        cov = tf.matmul(cov, scl)

        return cov

    # Networks forward pass
    @tf.function
    def forward(self, state):

        out = self.net.call(state)
        mu  = out[0]
        sg  = out[1]*self.sigma
        cr  = out[2]

        return mu, sg, cr
