# Generic imports
import gym
import numpy           as np
import multiprocessing as mp
import time

# Custom imports
from dragonfly.src.core.constants import *

###############################################
### A wrapper class for parallel environments
class par_envs:
    def __init__(self, env_name, n_cpu, path):

        # Init pipes and processes
        self.n_cpu = n_cpu
        self.pipes = []
        self.procs = []

        # Start environments
        for cpu in range(n_cpu):
            p_pipe, c_pipe = mp.Pipe()
            process        = mp.Process(target = worker,
                                        args   = (env_name, str(cpu), c_pipe, path))

            self.pipes.append(p_pipe)
            self.procs.append(process)

            process.daemon = True
            process.start()

        # Handle action and observation dimensions
        act_dim, obs_dim = self.get_dims()
        self.act_dim     = int(act_dim)
        self.obs_dim     = int(obs_dim)

        # Handle actions scaling
        self.act_min, self.act_max, self.act_norm = self.get_act_bounds()
        self.act_avg = 0.5*(self.act_max + self.act_min)
        self.act_rng = 0.5*(self.act_max - self.act_min)

    # Reset all environments
    def reset_all(self):

        # Send
        for p in self.pipes:
            p.send(('reset', None))

        # Receive and normalize
        results = np.array([])
        for p in self.pipes:
            obs     = p.recv()
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
                obs_array[cpu] = obs

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

    # Set cpu indices
    def set_cpus(self):

        # Send
        for cpu in range(self.n_cpu):
            self.pipes[cpu].send(('set_cpu', [cpu, self.n_cpu]))

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

        # Send
        for cpu in range(self.n_cpu):
            act = actions[cpu]
            if (self.act_norm):
                for i in range(self.act_dim):
                    act[i] = self.act_rng[i]*act[i] + self.act_avg[i]

            self.pipes[cpu].send(('step', act))

        # Receive
        nxt  = np.array([])
        rwd  = np.array([])
        done = np.array([], dtype=np.bool)

        for p in self.pipes:
            n, r, d = p.recv()
            nxt     = np.append(nxt, n)
            rwd     = np.append(rwd, r)
            done    = np.append(done, bool(d))

        nxt = np.reshape(nxt, (-1,self.obs_dim))

        return nxt, rwd, done

    # Unroll parallel environments until next update
    def unroll(self, agent, counter):

        # Send
        for cpu in range(self.n_cpu):
            self.pipes[cpu].send(('unroll', None))

        # Receive and normalize
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                obs            = self.pipes[cpu].recv()
                obs_array[cpu] = obs

# Target function for process
def worker(env_name, name, pipe, path):
    env = gym.make(env_name)
    try:
        while True:
            # Receive command
            command, data = pipe.recv()

            # Execute commands
            if command == 'reset':
                obs = env.reset()
                pipe.send(obs)

            if command == 'step':
                nxt, rwd, done, _ = env.step(data)
                pipe.send((nxt, rwd, done))

            if command == 'seed':
                env.seed(data)
                pipe.send(None)

            if command == 'render':
                pipe.send(env.render(mode='rgb_array'))

            if command == 'get_dims':
                # Discrete action space
                if (type(env.action_space).__name__ == "Discrete"):
                    act_dim = env.action_space.n
                # Continuous action space
                if (type(env.action_space).__name__ == "Box"):
                    act_dim = env.action_space.shape[0]
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
                else:
                    act_min  = 1.0
                    act_max  = 1.0
                    act_norm = False
                pipe.send((act_min, act_max, act_norm))

            if command == 'set_cpu':
                if hasattr(env, 'cpu'):
                    env.set_cpu(data[0], data[1])

            if command == 'close':
                pipe.send(None)
                break
    finally:
        env.close()
