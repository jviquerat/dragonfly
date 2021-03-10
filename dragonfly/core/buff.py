# Generic imports
import numpy      as np
import tensorflow as tf

###############################################
### Parallel buffer class, used to temporarily
### store data from parallel environments
class par_buff:
    def __init__(self, n_cpu, dim):
        self.n_cpu = n_cpu
        self.dim   = dim
        self.reset()

    def reset(self):
        self.buff = [np.array([]) for _ in range(self.n_cpu)]

    def append(self, vec):
        for cpu in range(self.n_cpu):
            self.buff[cpu] = np.append(self.buff[cpu], vec[cpu])

    def serialize(self):
        arr = np.array([])
        for cpu in range(self.n_cpu):
            arr = np.append(arr, self.buff[cpu])

        return np.reshape(arr, (-1,self.dim))

###############################################
### Local parallel buffer class, used to store
### data between two updates of the agent
class loc_buff:
    def __init__(self, n_cpu, obs_dim, act_dim):
        self.n_cpu   = n_cpu
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size    = 0
        self.reset()

    def reset(self):
        self.obs  = par_buff(self.n_cpu, self.obs_dim)
        self.nxt  = par_buff(self.n_cpu, self.obs_dim)
        self.act  = par_buff(self.n_cpu, self.act_dim)
        self.rwd  = par_buff(self.n_cpu, 1)
        self.trm  = par_buff(self.n_cpu, 1)
        self.size = 0

    def store(self, obs, nxt, act, rwd, trm):
        self.obs.append(obs)
        self.nxt.append(nxt)
        self.act.append(act)
        self.rwd.append(rwd)
        self.trm.append(trm)
        self.size += self.n_cpu

    def fix_trm_buffer(self):
        for cpu in range(self.n_cpu):
            if (self.trm.buff[cpu][-1] == 0):
                self.trm.buff[cpu][-1] = 2

    def serialize(self):
        obs = self.obs.serialize()
        nxt = self.nxt.serialize()
        act = self.act.serialize()
        rwd = self.rwd.serialize()
        trm = self.trm.serialize()

        return obs, nxt, act, rwd, trm

###############################################
### Global parallel buffer class, use to store
### all data since the beginning of learning
class glb_buff:
    def __init__(self, n_cpu, obs_dim, act_dim, n_buff, buff_size, btc_frac):
        self.n_cpu     = n_cpu
        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self.n_buff    = n_buff
        self.buff_size = buff_size
        self.btc_frac  = btc_frac
        self.size      = 0
        self.reset()

    def reset(self):
        self.obs  = np.empty([0,self.obs_dim])
        self.adv  = np.empty([0,1])
        self.tgt  = np.empty([0,1])
        self.act  = np.empty([0,self.act_dim])
        self.size = 0

    def store(self, obs, adv, tgt, act):
        self.obs = np.append(self.obs, obs, axis=0)
        self.adv = np.append(self.adv, adv, axis=0)
        self.tgt = np.append(self.tgt, tgt, axis=0)
        self.act = np.append(self.act, act, axis=0)

    def get(self):
        # Start/end indices
        end    = len(self.obs)
        start  = max(0,end - self.n_buff*self.buff_size)
        size   = end - start

        # Randomize batch
        sample = np.arange(start, end)
        np.random.shuffle(sample)

        # Get shuffled buffer
        obs = [self.obs[i] for i in sample]
        act = [self.act[i] for i in sample]
        adv = [self.adv[i] for i in sample]
        tgt = [self.tgt[i] for i in sample]

        # Reshape
        obs = tf.reshape(tf.cast(obs, tf.float32), [size, self.obs_dim])
        act = tf.reshape(tf.cast(act, tf.float32), [size, self.act_dim])
        adv = tf.reshape(tf.cast(adv, tf.float32), [size])
        tgt = tf.reshape(tf.cast(tgt, tf.float32), [size])

        return obs, act, adv, tgt
