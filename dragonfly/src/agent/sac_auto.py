# Custom imports
from dragonfly.src.agent.sac           import *
from dragonfly.src.optimizer.optimizer import opt_factory
from dragonfly.src.loss.alpha_sac      import alpha_sac
from dragonfly.src.utils.polyak        import polyak

###############################################
### SAC auto agent
class sac_auto(sac):
    def __init__(self, spaces, n_cpu, size, pms):
        super().__init__(spaces, n_cpu, size, pms)

        # Possible auto temperature
        self.alpha_pms   = self.alpha
        self.alpha_loss  = alpha_sac()
        self.tgt_entropy =-self.act_dim()

        self.monitor = False
        if (hasattr(self.alpha_pms, "monitor")): self.monitor = self.alpha_pms.monitor

    # Training
    def train(self, start, end):

        obs = self.data["obs"][start:end]
        nxt = self.data["nxt"][start:end]
        act = self.data["act"][start:end]
        rwd = self.data["rwd"][start:end]
        trm = self.data["trm"][start:end]

        size = end - start
        act  = self.p.reshape_actions(act)
        rwd  = tf.reshape(rwd, [size,-1])
        trm  = tf.reshape(trm, [size,-1])

        # Train q network
        self.q1.loss.train(obs, nxt, act, rwd, trm, self.gamma,
                           self.alpha, self.p, self.q1.net,
                           self.q1.tgt, self.q2.tgt, self.q1.opt)
        self.q2.loss.train(obs, nxt, act, rwd, trm, self.gamma,
                           self.alpha, self.p, self.q2.net,
                           self.q1.tgt, self.q2.tgt, self.q2.opt)

        # Train policy network with alpha update
        self.p.loss.train(obs, self.p, self.q1.net, self.q2.net,
                          self.alpha, self.p.opt)

        # Update alpha
        self.alpha_loss.train(obs, self.p, self.log_alpha,
                              self.tgt_entropy, self.alpha_opt)
        self.alpha = tf.math.exp(self.log_alpha)

        # Monitor if required
        if (self.monitor):
            with open("alpha", "a") as f:
                f.write(str(self.alpha.numpy())+"\n")

        # Update target networks
        self.polyak.average(self.q1.net, self.q1.tgt)
        self.polyak.average(self.q2.net, self.q2.tgt)

    # Reset
    def reset(self):

        super().reset()

        self.alpha       = self.alpha_pms.alpha_0
        self.log_alpha   = tf.Variable(tf.math.log(self.alpha))
        self.alpha_opt   = opt_factory.create(self.alpha_pms.optimizer.type,
                                              pms = self.alpha_pms.optimizer,
                                              grad_vars = [self.log_alpha])
