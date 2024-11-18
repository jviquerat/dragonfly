# Generic imports
import random
import numpy      as np
import tensorflow as tf

# Custom imports
from dragonfly.src.core.constants          import *
from dragonfly.src.policy.policy           import pol_factory
from dragonfly.src.value.value             import val_factory
from dragonfly.src.decay.decay             import decay_factory
from dragonfly.src.retrn.retrn             import retrn_factory
from dragonfly.src.termination.termination import termination_factory
from dragonfly.src.utils.buff              import buff, gbuff
from dragonfly.src.utils.timer             import timer
from dragonfly.src.utils.error             import error

###############################################
### Base agent
class base_agent():
    def __init__(self, spaces):
        self.spaces = spaces

    # Accessor
    def act_dim(self):
        return self.spaces.act_dim()

    # Accessor
    def true_act_dim(self):
        return self.spaces.true_act_dim()

    # Accessor
    def obs_dim(self):
        return self.spaces.obs_dim()

    # Accessor
    def obs_shape(self):
        return self.spaces.obs_shape()

    # Actions to execute before the inner training loop
    def pre_loop(self):

        self.buff.reset()

    # Save policy parameters
    def save_policy(self, filename):

        self.p.save(filename + '.weights.h5')

    # Load policy parameters
    def load_policy(self, folder):

        filename = folder + '/' + self.name + '.weights.h5'
        self.p.load(filename)

###############################################
### Base for on-policy agents
class base_agent_on_policy(base_agent):
    def __init__(self, spaces):
        super().__init__(spaces)

    # Create storage buffers
    def create_buffers(self):

        self.lnames = ["obs", "nxt", "act", "lgp", "rwd", "trm"]
        self.lsizes = [self.obs_dim(), self.obs_dim(), self.true_act_dim(), 1, 1, 1]
        self.buff   = buff(self.n_cpu, self.lnames, self.lsizes)

        self.gnames = ["obs", "act", "adv", "tgt", "lgp"]
        self.gsizes = [self.obs_dim(), self.true_act_dim(), 1, 1, 1]
        self.gbuff  = gbuff(self.mem_size, self.gnames, self.gsizes)

    # Get actions
    def actions(self, obs):

        # Get actions and associated log-prob
        self.timer_actions.tic()
        act, lgp = self.p.actions(obs)

        # Check for NaNs
        if (np.isnan(act).any()):
            error("a2c", "get_actions", "Detected NaN in generated actions")

        # Store log-prob
        self.buff.store(["lgp"], [lgp])

        self.timer_actions.toc()

        return act

    # Control (deterministic actions)
    def control(self, obs):

        return self.p.control(obs)

    # Finalize buffers before training
    def returns(self, obs, nxt, rwd, trm):

        # Get current and next values
        cval = self.v.values(obs)
        nval = self.v.values(nxt)

        # Compute advantages
        tgt, adv = self.retrn.compute(rwd, cval, nval, trm)

        return tgt, adv

    # Prepare training data
    def prepare_data(self, size):

        self.data = self.gbuff.get_buffers(self.gnames, size)
        lgt = len(self.data["obs"])

        return lgt, True

    # Actions to execute after the inner training loop
    def post_loop(self, style=None):

        # For buffer-style training, the last step of each buffer
        # must be bootstraped to mimic a continuing episode
        if ((style == "buffer") and (self.term.type == "bootstrap")):
            for i in range(self.n_cpu):
                done = (self.buff.data["trm"].buff[i][-1] == 0.0)
                if (not done):
                    self.buff.data["trm"].buff[i][-1] = 2.0

        names = ["obs", "nxt", "act", "lgp", "rwd", "trm"]
        data  = self.buff.serialize(names)
        gobs, gnxt, gact, glgp, grwd, gtrm = (data[name] for name in names)
        gtgt, gadv = self.returns(gobs, gnxt, grwd, gtrm)

        self.gbuff.store(self.gnames, [gobs, gact, gadv, gtgt, glgp])

    # Agent reset
    def reset(self):

        self.p.reset()
        self.v.reset()
        self.buff.reset()
        self.gbuff.reset()

    # Store transition
    def store(self, obs, nxt, act, rwd, dne, trc):

        trm = self.term.terminate(dne, trc)
        self.buff.store(["obs", "nxt", "act", "rwd", "trm"],
                        [ obs,   nxt,   act,   rwd,   trm ])

###############################################
### Base for off-policy agents
class base_agent_off_policy(base_agent):
    def __init__(self, spaces):
        super().__init__(spaces)

    # Create storage buffers
    def create_buffers(self):

        self.names = ["obs", "nxt", "act", "rwd", "trm"]
        self.sizes = [self.obs_dim(), self.obs_dim(), self.true_act_dim(), 1, 1]
        self.buff  = buff(self.n_cpu, self.names, self.sizes)
        self.gbuff = gbuff(self.mem_size, self.names, self.sizes)

    # Control (deterministic actions)
    def control(self, obs):

        return self.p.control(obs)

    # Prepare training data
    def prepare_data(self, size):

        # No update if buffer is not full enough
        lgt = self.gbuff.length()
        if (lgt < min(size, self.n_filling)): return lgt, False

        self.data = self.gbuff.get_batches(self.names, size)
        return lgt, True

    # Actions to execute after the inner training loop
    def post_loop(self):

        data = self.buff.serialize(self.names)
        gobs, gnxt, gact, grwd, gtrm = (data[name] for name in self.names)

        self.gbuff.store(self.names, [gobs, gnxt, gact, grwd, gtrm])

    # Store transition
    def store(self, obs, nxt, act, rwd, dne, trc):

        trm = self.term.terminate(dne, trc)
        self.buff.store(self.names, [obs, nxt, act, rwd, trm])
