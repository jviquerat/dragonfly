# Generic imports
import random
import numpy as np

# Custom imports
from dragonfly.src.policy.policy           import *
from dragonfly.src.value.value             import *
from dragonfly.src.decay.decay             import *
from dragonfly.src.retrn.retrn             import *
from dragonfly.src.srl.srl                 import *
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

    # Initialize srl
    def init_srl(self, pms, obs_dim, buff_size):

        # Check inputs
        if not hasattr(pms, "srl"):
            pms.srl  = None
            srl_type = "dummy"
        else: srl_type = pms.srl.type

        # Create srl
        self.srl = srl_factory.create(srl_type,
                                      obs_dim   = obs_dim,
                                      buff_size = buff_size,
                                      pms       = pms.srl)

    # Pre-process observations using srl
    def process_obs(self, obs):

        return self.srl.process(obs)

    # Reset
    def reset(self):
        raise NotImplementedError

    # Actions to execute before the inner training loop
    def pre_loop(self):

        self.buff.reset()

    # Save policy parameters
    def save_policy(self, filename):

        self.p.save(filename)

    # Load policy parameters
    def load_policy(self, filename):

        self.p.load(filename)

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

        if hasattr(self, "srl"): pobs = super().process_obs(obs)
        else: pobs = obs

        # Get actions and associated log-prob
        self.timer_actions.tic()
        act, lgp = self.p.actions(pobs)

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

        if hasattr(self, "srl"): pobs = super().process_obs(obs)
        else: pobs = obs

        return self.p.control(pobs)

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

        if hasattr(self, "srl"):
            pgobs = super().process_obs(gobs)
            pgnxt = super().process_obs(gnxt)
        else:
            pgobs = gobs
            pgnxt = gnxt

        gtgt, gadv = self.returns(pgobs, pgnxt, grwd, gtrm)

        self.gbuff.store(self.gnames, [gobs, gact, gadv, gtgt, glgp])

    # Agent reset
    def reset(self):

        self.p.reset()
        self.v.reset()
        self.buff.reset()
        self.gbuff.reset()
        if hasattr(self, "srl"):
            self.srl.reset()

    # Store transition
    def store(self, obs, nxt, act, rwd, dne, trc):

        trm = self.term.terminate(dne, trc)
        self.buff.store(["obs", "nxt", "act", "rwd", "trm"],
                        [ obs,   nxt,   act,   rwd,   trm ])

        # Store in SRL buffer
        if hasattr(self, "srl"): self.srl.store_obs(obs)

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
