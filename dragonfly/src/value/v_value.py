# Custom imports
from dragonfly.src.value.base import *

###############################################
### v_value class
### obs_dim : input  dimension
### pms     : parameters
class v_value(base_value):
    def __init__(self, obs_dim, pms):

        # Fill structure
        self.dim = 1
        self.obs_dim = obs_dim

        # Define and init network
        # Force linear activation, as this is v-value network
        pms.network.fnl_actv = "linear"
        self.net = net_factory.create(pms.network.type,
                                      inp_dim = obs_dim,
                                      out_dim = [self.dim],
                                      pms     = pms.network)

        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainables)

        # Define loss
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

    # Get values
    def get_values(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Predict values
        values = np.array(self.call_net(obs))
        values = np.reshape(values, (-1,1))

        return values

    # Call loss for training
    def train(self, obs, tgt, size):

        tgt = tf.reshape(tgt, [-1])
        return self.loss.train(obs, tgt, size, self)
