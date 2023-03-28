# Generic imports
import os
import sys
import gym
import numpy as np
import time

# Custom imports
from dragonfly.src.envs.worker    import *
from dragonfly.src.core.constants import *
from dragonfly.src.utils.timer    import *

###############################################
### A wrapper class for parallel environments
class environments:
    def __init__(self, path, pms):

        # Default parameters
        self.name      = pms.name
        self.act_norm  = True
        self.obs_norm  = True
        self.obs_clip  = False
        self.obs_noise = False
        self.args      = None

        if hasattr(pms, "args"):      self.args     = pms.args
        if hasattr(pms, "act_norm"):  self.act_norm = pms.act_norm
        if hasattr(pms, "obs_norm"):  self.obs_norm = pms.obs_norm
        if hasattr(pms, "obs_clip"):  self.obs_clip = pms.obs_clip
        if hasattr(pms, "obs_noise"): self.obs_norm = pms.obs_noise

        # Generate workers
        self.worker = worker(self.name, self.args, mpi.rank, path)

        # Set all slaves to wait for instructions
        if (mpi.rank != 0):
            self.worker.work()

        # Desactivate act_norm for discrete environments
        if (self.get_action_type() == "discrete"):
            self.act_norm = False

        # Possibly set obs max by hand
        self.manual_obs_max = 1.0
        if hasattr(pms, "obs_max"): self.manual_obs_max = pms.obs_max

        # Handle action and observation dimensions
        self.get_dims()

        # Handle actions bounds
        self.get_act_bounds()
        self.act_avg = 0.5*(self.act_max + self.act_min)
        self.act_rng = 0.5*(self.act_max - self.act_min)

        # Handle observations bounds
        self.get_obs_bounds()
        self.obs_min = np.where(self.obs_min < -def_obs_max,
                                -self.manual_obs_max,
                                self.obs_min)
        self.obs_max = np.where(self.obs_max >  def_obs_max,
                                self.manual_obs_max,
                                self.obs_max)
        self.obs_avg = 0.5*(self.obs_max + self.obs_min)
        self.obs_rng = 0.5*(self.obs_max - self.obs_min)

        # Initialize timer
        self.timer_env = timer("env      ")

    # Take one step in all environments
    def step(self, actions):

        self.timer_env.tic()

        # Send
        data = [('step', None)]*mpi.size
        for p in range(mpi.size):
            act = actions[p]
            if (self.act_norm):
                act = np.clip(act,-1.0,1.0)
                for i in range(self.act_dim):
                    act[i] = self.act_rng[i]*act[i] + self.act_avg[i]
            data[p] = ('step', act)
        mpi.comm.scatter(data, root=0)

        # Main process executing
        n, r, d, t = self.worker.step(data[0][1])

        # Receive
        nxt   = np.empty((mpi.size, self.obs_dim))
        rwd   = np.empty((mpi.size))
        done  = np.empty((mpi.size))
        trunc = np.empty((mpi.size))

        data  = mpi.comm.gather((n, r, d, t), root=0)

        for p in range(mpi.size):
            vals       = data[p]
            n, r, d, t = vals[0], vals[1], vals[2], vals[3]
            n          = self.process_obs(n)
            nxt  [p]   = np.reshape(n, (self.obs_dim))
            rwd  [p]   = r
            done [p]   = bool(d)
            trunc[p]   = bool(t)

        self.timer_env.toc()

        return nxt, rwd, done, trunc

    # Reset all environments
    def reset_all(self):

        # Send
        data = [('reset', True) for i in range(mpi.size)]
        mpi.comm.scatter(data, root=0)

        # Main process executing
        obs = self.worker.reset(data[0][1])

        # Receive and normalize
        results = np.empty((mpi.size, self.obs_dim))
        data    = mpi.comm.gather((obs), root=0)
        for p in range(mpi.size):
            results[p] = self.process_obs(data[p])

        return results

    # Reset based on a done array
    def reset(self, done, obs_array):

        # Send
        data = [('reset', True) for i in range(mpi.size)]
        for p in range(mpi.size):
            data[p] = ('reset', done[p])
        mpi.comm.scatter(data, root=0)

        # Main process executing
        obs = self.worker.reset(data[0][1])

        # Receive and normalize
        data = mpi.comm.gather(obs, root=0)
        for p in range(mpi.size):
            if (done[p]):
                obs_array[p] = self.process_obs(data[p])

    # Check action type
    def get_action_type(self):

        # Discrete action type
        if (type(self.worker.env.action_space).__name__ == "Discrete"):
            t = "discrete"
        # Continuous action type
        if (type(self.worker.env.action_space).__name__ == "Box"):
            t = "continuous"

        return t

    # Get environment dimensions
    def get_dims(self):

        # Discrete action space
        if (type(self.worker.env.action_space).__name__ == "Discrete"):
            self.act_dim = int(self.worker.env.action_space.n)
        # Continuous action space
        if (type(self.worker.env.action_space).__name__ == "Box"):
            self.act_dim = int(self.worker.env.action_space.shape[0])
        # Discrete observation space
        if (type(self.worker.env.observation_space).__name__ == "Discrete"):
            self.obs_dim = int(self.worker.env.observation_space.n)
        # Continuous observation space
        if (type(self.worker.env.observation_space).__name__ == "Box"):
            self.obs_dim = int(self.worker.env.observation_space.shape[0])

    # Get action boundaries
    def get_act_bounds(self):

        # Continuous action space
        if (type(self.worker.env.action_space).__name__ == "Box"):
            self.act_min  = self.worker.env.action_space.low
            self.act_max  = self.worker.env.action_space.high
        # Discrete action space
        if (type(self.worker.env.action_space).__name__ == "Discrete"):
            self.act_min  = 1.0
            self.act_max  = 1.0

    # Get observations boundaries
    def get_obs_bounds(self):

        # Continuous observation space
        if (type(self.worker.env.observation_space).__name__ == "Box"):
            self.obs_min  = self.worker.env.observation_space.low
            self.obs_max  = self.worker.env.observation_space.high
        # Discrete observation space ?
        else:
            self.obs_min  = 1.0
            self.obs_max  = 1.0

    # Render environment
    def render(self, render):

        # Not all environments will render simultaneously
        # We use a list to store those that render and those that don't
        rnd = [[] for _ in range(mpi.size)]

        # Send
        data = [('render',False) for i in range(mpi.size)]
        for cpu in range(mpi.size):
            if (render[cpu]):
                data[cpu] = ('render',True)
        mpi.comm.scatter(data, root=0)

        # Main process executing
        r = self.worker.render(data[0][1])

        # Receive
        data = mpi.comm.gather(r, root=0)
        for cpu in range(mpi.size):
            if (render[cpu]):
                rnd[cpu] = data[cpu]

        return rnd

    # Render environment
    def render_single(self, cpu):

        # Send
        data      = [('render',False) for i in range(mpi.size)]
        data[cpu] = ('render',True)
        mpi.comm.scatter(data, root=0)

        # Main process executing
        r = self.worker.render(data[0][1])

        # Receive
        data = mpi.comm.gather(r, root=0)
        rgb = data[cpu]

        return rgb

    # Close
    def close(self):

        data = [('close',None) for i in range(mpi.size)]
        data = mpi.comm.scatter(data, root=0)

        # Main process executing
        self.worker.close()

    # Set control to true if possible
    def set_control(self):

        if (hasattr(self.worker.env, 'control')):
            self.worker.env.control = True

    # Process observations
    def process_obs(self, obs):
        if (self.obs_clip):
            obs = self.clip_obs(obs)
        if (self.obs_norm):
            obs = self.norm_obs(obs)
        if (self.obs_noise > 0.0):
            obs = self.noise_obs(obs)

        return obs

    # Clip observations
    def clip_obs(self, obs):

        for i in range(self.obs_dim):
            obs[i] = np.clip(obs[i], self.obs_min[i], self.obs_max[i])

        return obs

    # Normalize observations
    def norm_obs(self, obs):

        obs -= self.obs_avg
        obs /= self.obs_rng

        return obs

    # Add noise to observations
    def noise_obs(self, obs):

        noise = np.random.normal(0.0, self.obs_noise, self.obs_dim)
        for i in range(self.obs_dim):
            obs[i] += noise[i]

        return obs
