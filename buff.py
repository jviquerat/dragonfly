# Generic imports
import numpy as np

# ###############################################
# ### Parallel buffer class
# class par_buff:
#     def __init__(self, n_cpu, dim):
#         self.n_cpu = n_cpu
#         self.dim   = dim
#         self.reset()

#     def reset(self):
#         self.buff = [np.array([]) for _ in range(self.n_cpu)]
#         #for cpu in range(self.n_cpu):
#         #   self.buff.append(np.empty(0,self.dim))

#     def append(self, vec):
#         for cpu in range(self.n_cpu):
#             self.buff[cpu] = np.append(self.buff[cpu], vec[cpu])

###############################################
### Local parallel buffer class
class loc_buff:
    def __init__(self, n_cpu, obs_dim, act_dim):
        self.n_cpu   = n_cpu
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        self.obs  = np.array([])
        self.nxt  = np.array([])
        self.act  = np.array([])
        self.rwd  = np.array([])
        self.trm  = np.array([])
        self.size = 0

    def store(self, obs, nxt, act, rwd, trm):
        self.obs   = np.append(self.obs, obs)
        self.nxt   = np.append(self.nxt, nxt)
        self.act   = np.append(self.act, act)
        self.rwd   = np.append(self.rwd, rwd)
        self.trm   = np.append(self.trm, trm)
        self.size += 1

    def reshape(self):
        self.obs = np.reshape(self.obs,(-1,self.obs_dim))
        self.nxt = np.reshape(self.nxt,(-1,self.obs_dim))
        self.act = np.reshape(self.act,(-1,self.act_dim))
        self.rwd = np.reshape(self.rwd,(-1,1))
        self.trm = np.reshape(self.trm,(-1,1))

# ###############################################
# ### Global parallel buffer class
# class glb_buff:
#     def __init__(self, n_cpu, obs_dim, act_dim):
#         self.n_cpu   = n_cpu
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.reset()

#     def reset(self):
#         self.obs = par_buff(self.n_cpu, self.obs_dim)
#         self.adv = par_buff(self.n_cpu, 1)
#         self.tgt = par_buff(self.n_cpu, 1)
#         self.act = par_buff(self.n_cpu, self.act_dim)

#     def store(self, obs, adv, tgt, act):
#         self.obs.append(obs)
#         self.adv.append(adv)
#         self.tgt.append(tgt)
#         self.act.append(act)
