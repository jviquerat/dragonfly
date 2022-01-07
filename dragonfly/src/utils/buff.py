# Generic imports
import math
import numpy      as np
import tensorflow as tf

###############################################
### Parallel buffer class, used to temporarily
### store data from parallel environments
### n_cpu : nb of parallel environments
### dim   : dimension of array
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

    def length(self):

        return len(self.buff[0])

    def serialize(self):

        arr = np.array([])
        for cpu in range(self.n_cpu):
            arr = np.append(arr, self.buff[cpu])

        return np.reshape(arr, (-1,self.dim))

###############################################
### Local parallel buffer class, used to store
### data between two updates of the agent
### n_cpu     : nb of parallel environments
### obs_dim   : dimension of observations
### act_dim   : dimension of actions
class loc_buff:
    def __init__(self, n_cpu, obs_dim, act_dim):

        self.n_cpu   = n_cpu
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):

        self.obs = par_buff(self.n_cpu, self.obs_dim)
        self.nxt = par_buff(self.n_cpu, self.obs_dim)
        self.act = par_buff(self.n_cpu, self.act_dim)
        self.lgp = par_buff(self.n_cpu, 1)
        self.rwd = par_buff(self.n_cpu, 1)
        self.dne = par_buff(self.n_cpu, 1)
        self.stp = par_buff(self.n_cpu, 1)

        self.trm = par_buff(self.n_cpu, 1)
        self.bts = par_buff(self.n_cpu, 1)

    def store(self, obs, nxt, act, lgp, rwd, dne, stp):

        self.obs.append(obs)
        self.nxt.append(nxt)
        self.act.append(act)
        self.lgp.append(lgp)
        self.rwd.append(rwd)
        self.dne.append(dne)
        self.stp.append(stp)

    def store_terminal(self, trm, bts):

        self.trm.append(trm)
        self.bts.append(bts)

    def size(self):

        return self.rwd.length()*self.n_cpu

    def length(self):

        return self.rwd.length()

    def serialize(self):

        obs = self.obs.serialize()
        nxt = self.nxt.serialize()
        act = self.act.serialize()
        lgp = self.lgp.serialize()
        rwd = self.rwd.serialize()
        trm = self.trm.serialize()
        bts = self.bts.serialize()

        return obs, nxt, act, lgp, rwd, trm, bts

###############################################
### Global parallel buffer class, used to store
### all data since the beginning of learning
### It is also responsible for providing buffer
### indices during training procedure
### n_cpu     : nb of parallel environments
### obs_dim   : dimension of observations
### act_dim   : dimension of actions
### n_buff    : nb of buffers from history to return
### buff_size : max buffer size
### btc_frac  : relative size of a batch compared to buffer (in [0,1])
class glb_buff:
    def __init__(self, n_cpu, obs_dim, act_dim):

        self.n_cpu     = n_cpu
        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self.reset()

    def reset(self):

        self.obs  = np.empty([0,self.obs_dim])
        self.act  = np.empty([0,self.act_dim])
        self.adv  = np.empty([0,1])
        self.tgt  = np.empty([0,1])
        self.lgp  = np.empty([0,1])

    def store(self, obs, adv, tgt, act, lgp):

        self.obs = np.append(self.obs, obs, axis=0)
        self.adv = np.append(self.adv, adv, axis=0)
        self.tgt = np.append(self.tgt, tgt, axis=0)
        self.act = np.append(self.act, act, axis=0)
        self.lgp = np.append(self.lgp, lgp, axis=0)

    def get_buffers(self, size):

        # Start/end indices
        end    = len(self.obs)
        start  = max(0,end - size)
        size   = end - start

        # Randomize batch
        sample = np.arange(start, end)
        np.random.shuffle(sample)

        # Get shuffled buffer
        obs = [self.obs[i] for i in sample]
        act = [self.act[i] for i in sample]
        adv = [self.adv[i] for i in sample]
        tgt = [self.tgt[i] for i in sample]
        lgp = [self.lgp[i] for i in sample]

        # Reshape
        obs = tf.reshape(tf.cast(obs, tf.float32), [size, self.obs_dim])
        act = tf.reshape(tf.cast(act, tf.float32), [size, self.act_dim])
        adv = tf.reshape(tf.cast(adv, tf.float32), [size])
        tgt = tf.reshape(tf.cast(tgt, tf.float32), [size])
        lgp = tf.reshape(tf.cast(lgp, tf.float32), [size])

        return obs, act, adv, tgt, lgp
