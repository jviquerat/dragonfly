# Custom imports
from dragonfly.src.value.base import *

###############################################
### q_value class
### obs_dim : input  dimension
### act_dim : action dimension
### pms     : parameters
class q_value(base_value):
    def __init__(self, obs_dim, act_dim, pms):
        super(q_value, self).__init__()

        # Fill structure
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # Define and init network
        if (pms.network.heads.final[0] != "linear"):
            warning("q_value", "__init__",
                    "Chosen final activation for q_value is not linear")

        self.net = net_factory.create(pms.network.type,
                                      inp_dim = self.obs_dim,
                                      out_dim = [self.act_dim],
                                      pms     = pms.network)

        # Define trainables
        self.trainables = self.net.trainable_weights

        # Define optimizer
        self.opt = opt_factory.create(pms.optimizer.type,
                                      pms       = pms.optimizer,
                                      grad_vars = self.trainables)

        # Define loss
        if (pms.loss.type != "mse_q"):
            warning("q_value", "__init__",
                    "Chosen loss for q_value is not mse_q")
        self.loss = loss_factory.create(pms.loss.type,
                                        pms = pms.loss)

    # Get values
    def get_values(self, obs):

        # Cast
        obs = tf.cast(obs, tf.float32)

        # Predict values
        values = np.array(self.call_net(obs))
        values = np.reshape(values, (-1,self.act_dim))

        return values

    # Call loss for training
    def train(self, obs, act, tgt, size):

        return self.loss.train(obs, act, tgt, size, self)
