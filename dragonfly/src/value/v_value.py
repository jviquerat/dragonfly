# Custom imports
from dragonfly.src.value.base import *

###############################################
### v_value class
### inp_dim : input dimension
### pms     : parameters
class v_value(base_value):
    def __init__(self, inp_dim, pms):

        # Fill structure
        self.inp_dim = inp_dim
        self.out_dim = 1

        # Define and init network
        if (pms.network.heads.final[0] != "linear"):
            warning("v_value", "__init__",
                    "Chosen final activation for v_value is not linear")

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
    def train(self, obs, tgt, size):

        tgt = tf.reshape(tgt, [-1])
        return self.loss.train(obs, tgt, size, self)
