# Generic imports
import random
import numpy as np

# Custom imports
from dragonfly.src.policy.policy           import *
from dragonfly.src.value.value             import *
from dragonfly.src.decay.decay             import *
from dragonfly.src.retrn.retrn             import *
from dragonfly.src.core.constants          import *
from dragonfly.src.termination.termination import *
from dragonfly.src.utils.buff              import *
from dragonfly.src.utils.timer             import *
from dragonfly.src.utils.error             import *

###############################################
### Base agent
class base_agent():
    def __init__(self):
        pass

    # Get actions
    def actions(self, obs):
        raise NotImplementedError

    # Reset
    def reset(self):
        raise NotImplementedError

    # Actions to execute before the inner training loop
    def pre_loop(self):

        self.buff.reset()

    # Save
    def save(self, filename):
        raise NotImplementedError

    # Load
    def load(self, filename):
        raise NotImplementedError

###############################################
### Base for on-policy agents
class base_agent_on_policy(base_agent):
    def __init__(self):
        pass

    # Create storage buffers
    def create_buffers(self):

        self.lnames = ["obs", "nxt", "act", "lgp", "rwd", "trm"]
        self.lsizes = [self.obs_dim, self.obs_dim, self.pol_dim, 1, 1, 1]
        self.buff   = buff(self.n_cpu, self.lnames, self.lsizes)

        self.gnames = ["obs", "act", "adv", "tgt", "lgp"]
        self.gsizes = [self.obs_dim, self.pol_dim, 1, 1, 1]
        self.gbuff  = gbuff(self.size, self.gnames, self.gsizes)

    # Get actions
    def actions(self, obs):

        # Get actions and associated log-prob
        self.timer_actions.tic()
        act, lgp = self.p_net.actions(obs)

        # Check for NaNs
        if (np.isnan(act).any()):
            error("a2c", "get_actions",
                  "Detected NaN in generated actions")

        # Store log-prob
        self.buff.store(["lgp"], [lgp])

        self.timer_actions.toc()

        return act

    # Control (deterministic actions)
    def control(self, obs):

        return self.p_net.control(obs)

    # Finalize buffers before training
    def returns(self, obs, nxt, rwd, trm):

        # Get current and next values
        cval = self.v_net.values(obs)
        nval = self.v_net.values(nxt)

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

        self.p_net.reset()
        self.v_net.reset()
        self.buff.reset()
        self.gbuff.reset()

    # Store transition
    def store(self, obs, nxt, act, rwd, dne, trc):

        trm = self.term.terminate(dne, trc)
        self.buff.store(["obs", "nxt", "act", "rwd", "trm"],
                        [ obs,   nxt,   act,   rwd,   trm ])

    # Save agent parameters
    def save(self, filename):

        self.p_net.save(filename)

    # Load agent parameters
    def load(self, filename):

        self.p_net.load(filename)


###############################################
### Base for off-policy agents
class base_agent_off_policy(base_agent):
    def __init__(self):
        pass

    # Create storage buffers
    def create_buffers(self, act_dim):

        self.names = ["obs", "nxt", "act", "rwd", "trm"]
        self.sizes = [self.obs_dim, self.obs_dim, act_dim, 1, 1]
        self.buff  = buff(self.n_cpu, self.names, self.sizes)
        self.gbuff = gbuff(self.mem_size, self.names, self.sizes)

    # Control (deterministic actions)
    def control(self, obs):

        return self.p_net.control(obs)

    # Prepare training data
    def prepare_data(self, size):

        # No update if buffer is not full enough
        lgt = self.gbuff.length()
        if (lgt < max(size, self.n_filling)): return lgt, False

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

     # Save agent parameters
    def save(self, filename):

        self.p_net.save(filename)

    # Load agent parameters
    def load(self, filename):

        self.p_net.load(filename)
