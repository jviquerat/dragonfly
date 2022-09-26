# Generic imports
import os
import sys
import gym
import numpy           as np
import multiprocessing as mp
import time

# Custom imports
from dragonfly.src.core.constants import *
from dragonfly.src.utils.timer    import *

# Set warning levels from gym
gym.logger.set_level(40)

###############################################
### A wrapper class for parallel environments
class par_envs:
    def __init__(self, n_cpu, path, pms):

        # Init pipes and processes
        self.name      = pms.name
        self.act_norm  = True
        self.obs_clip  = pms.obs_clip
        self.obs_norm  = pms.obs_norm
        self.obs_noise = pms.obs_noise
        self.n_cpu     = n_cpu
        self.pipes     = []
        self.procs     = []

        # Possibly set obs max by hand
        self.manual_obs_max = 1.0
        if hasattr(pms, "obs_max"): self.manual_obs_max = pms.obs_max

        # Start environments
        for cpu in range(n_cpu):
            p_pipe, c_pipe = mp.Pipe()
            process        = mp.Process(target = worker,
                                        args   = (self.name, str(cpu),
                                                  c_pipe, path))

            self.pipes.append(p_pipe)
            self.procs.append(process)

            process.daemon = True
            process.start()

        # Handle action and observation dimensions
        act_dim, obs_dim = self.get_dims()
        self.act_dim     = int(act_dim)
        self.obs_dim     = int(obs_dim)

        # Handle actions scaling
        self.act_min, self.act_max, act_norm = self.get_act_bounds()
        if ((not act_norm) and self.act_norm): self.act_norm = False
        self.act_avg = 0.5*(self.act_max + self.act_min)
        self.act_rng = 0.5*(self.act_max - self.act_min)

        self.obs_min, self.obs_max, obs_norm = self.get_obs_bounds()
        if ((not obs_norm) and self.obs_norm): self.obs_norm = False
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

    # Reset all environments
    def reset_all(self):

        # Send
        for p in self.pipes:
            p.send(('reset', None))

        # Receive and normalize
        results = np.array([])
        for p in self.pipes:
            obs     = p.recv()
            obs     = self.process_obs(obs)
            results = np.append(results, obs)

        return np.reshape(results, (-1,self.obs_dim))

    # Reset based on a done array
    def reset(self, done, obs_array):

        # Send
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                self.pipes[cpu].send(('reset', None))

        # Receive and normalize
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                obs            = self.pipes[cpu].recv()
                obs            = self.process_obs(obs)
                obs_array[cpu] = obs

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

        for i in range(self.obs_dim):
            obs[i] = (obs[i] - self.obs_avg[i])/self.obs_rng[i]

        return obs

    # Add noise to observations
    def noise_obs(self, obs):

        noise = np.random.normal(0.0, self.obs_noise, self.obs_dim)
        for i in range(self.obs_dim):
            obs[i] += noise[i]

        return obs

    # Check action type
    def get_action_type(self):

        self.pipes[0].send(('get_action_type', None))
        t = self.pipes[0].recv()
        return t

    # Get environment dimensions
    def get_dims(self):

        # Send
        self.pipes[0].send(('get_dims', None))

        # Receive
        results = np.array([])
        results = np.append(results, self.pipes[0].recv())

        return results

    # Get action boundaries
    def get_act_bounds(self):

        # Send
        self.pipes[0].send(('get_act_bounds', None))

        # Receive
        act_min, act_max, act_norm = self.pipes[0].recv()

        return act_min, act_max, act_norm

    # Get observations boundaries
    def get_obs_bounds(self):

        # Send
        self.pipes[0].send(('get_obs_bounds', None))

        # Receive
        obs_min, obs_max, obs_norm = self.pipes[0].recv()

        return obs_min, obs_max, obs_norm

    # Render environment
    def render(self, render):

        # Not all environments will render simultaneously
        # We use a list to store those that render and those that don't
        rnd = [[] for _ in range(self.n_cpu)]

        # Send
        for cpu in range(self.n_cpu):
            if (render[cpu]):
                self.pipes[cpu].send(('render', None))

        # Receive
        for cpu in range(self.n_cpu):
            if (render[cpu]):
                rnd[cpu] = self.pipes[cpu].recv()

        return rnd

    # Render environment
    def render_single(self, cpu):

        # Send
        self.pipes[cpu].send(('render', None))

        # Receive
        rgb = self.pipes[cpu].recv()

        return rgb

    # Close
    def close(self):

        # Close all envs
        for p in self.pipes:
            p.send(('close', None))
        for p in self.procs:
            p.terminate()
            p.join()

    # Take one step in all environments
    def step(self, actions):

        self.timer_env.tic()

        # Send
        for cpu in range(self.n_cpu):
            act = actions[cpu]
            if (self.act_norm):
                act = np.clip(act,-1.0,1.0)
                for i in range(self.act_dim):
                    act[i] = self.act_rng[i]*act[i] + self.act_avg[i]

            self.pipes[cpu].send(('step', act))

        # Receive
        nxt   = np.array([])
        rwd   = np.array([])
        done  = np.array([], dtype=bool)
        trunc = np.array([], dtype=bool)

        for p in self.pipes:
            n, r, d, t = p.recv()
            n          = self.process_obs(n)
            nxt        = np.append(nxt,   n)
            rwd        = np.append(rwd,   r)
            done       = np.append(done,  bool(d))
            trunc      = np.append(trunc, bool(t))

        nxt = np.reshape(nxt, (-1,self.obs_dim))

        self.timer_env.toc()

        return nxt, rwd, done, trunc

    # Set control to true if possible
    def set_control(self):

        self.pipes[0].send(('set_control', None))

# Target function for process
def worker(env_name, cpu, pipe, path):

    # Build environment
    try:
        env = gym.make(env_name, render_mode="rgb_array")
    except:
        sys.path.append(path)
        module    = __import__(env_name)
        env_build = getattr(module, env_name)
        try:
            env = env_build(cpu)
        except:
            env = env_build()

    # Run
    try:
        while True:
            # Receive command
            command, data = pipe.recv()

            # Execute commands
            if command == 'reset':
                obs, _ = env.reset()
                pipe.send(obs)

            if command == 'step':
                nxt, rwd, done, trunc, _ = env.step(data)
                if ((not done) and trunc): done = True
                pipe.send((nxt, rwd, done, trunc))

            if command == 'seed':
                env.seed(data)
                pipe.send(None)

            if command == 'render':
                pipe.send(env.render())

            if command == "get_action_type":
                if (type(env.action_space).__name__ == "Discrete"):
                    pipe.send("discrete")
                if (type(env.action_space).__name__ == "Box"):
                    pipe.send("continuous")

            if command == 'get_dims':
                # Discrete action space
                if (type(env.action_space).__name__ == "Discrete"):
                    act_dim = env.action_space.n
                # Continuous action space
                if (type(env.action_space).__name__ == "Box"):
                    act_dim = env.action_space.shape[0]
                # Discrete observation space
                if (type(env.observation_space).__name__ == "Discrete"):
                    obs_dim = env.observation_space.n
                # Continuous observation space
                if (type(env.observation_space).__name__ == "Box"):
                    obs_dim = env.observation_space.shape[0]
                pipe.send((act_dim, obs_dim))

            if command == 'get_act_bounds':
                # Continuous action space
                if (type(env.action_space).__name__ == "Box"):
                    act_min  = env.action_space.low
                    act_max  = env.action_space.high
                    act_norm = True
                if (type(env.action_space).__name__ == "Discrete"):
                    act_min  = 1.0
                    act_max  = 1.0
                    act_norm = False
                pipe.send((act_min, act_max, act_norm))

            if (command == 'get_obs_bounds'):
                # Continuous observation space
                if (type(env.observation_space).__name__ == "Box"):
                    obs_min  = env.observation_space.low
                    obs_max  = env.observation_space.high
                    obs_norm = True
                else:
                    obs_min  = 1.0
                    obs_max  = 1.0
                    obs_norm = False
                pipe.send((obs_min, obs_max, obs_norm))

            if command == 'close':
                pipe.send(None)
                break

            if command == 'set_control':
                if (hasattr(env,'control')):
                    env.control = True
    finally:
        env.close()
