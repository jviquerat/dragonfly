# Custom imports
from dragonfly.src.value.base import *

###############################################
### q_value class
### inp_dim : input  dimension
### out_dim : output dimension
### pms     : parameters
class q_value(base_value):
    def __init__(self, inp_dim, out_dim, pms):

        # Fill structure
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Define and init network
        if (pms.network.heads.final[0] != "linear"):
            warning("q_value", "__init__",
                    "Chosen final activation for q_value is not linear")

        self.net = net_factory.create(pms.network.type,
                                      inp_dim = self.inp_dim,
                                      out_dim = [self.out_dim],
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
    def values(self, x):

        # Cast
        x = tf.cast(x, tf.float32)

        # Predict values
        v = np.array(self.forward(x))
        v = np.reshape(v, (-1,self.out_dim))

        return v

    # Call loss for training
    def train(self, obs, act, tgt, size):

        tgt = tf.reshape(tgt, [size,-1])
        act = tf.reshape(act, [size,-1])
        act = tf.cast(act, tf.int32)
        return self.loss.train(obs, act, tgt, size, self)
