# Generic imports
import gym
import numpy           as np
import multiprocessing as mp
import time

###############################################
### A wrapper class for parallel environments
class par_envs:
    def __init__(self, env_name, n_cpu, path):

        # Init pipes and processes
        self.n_cpu   = n_cpu
        self.p_pipes = []
        self.proc    = []

        # Start environments
        for cpu in range(n_cpu):
            p_pipe, c_pipe = mp.Pipe()
            name           = str(cpu)
            process        = mp.Process(target = worker,
                                        name   = name,
                                        args   = (env_name, name,
                                                  c_pipe, p_pipe, path))

            self.p_pipes.append(p_pipe)
            self.proc.append(process)

            process.daemon = True
            process.start()
            c_pipe.close()

        # Handle action and observation dimensions
        act_dim, obs_dim = self.get_dims()
        self.act_dim = int(act_dim)
        self.obs_dim = int(obs_dim)
        self.obs_min, self.obs_max = self.get_obs_bounds()
        self.act_min, self.act_max = self.get_act_bounds()

        # Set cpu indices
        self.set_cpus()

    # Reset all environments
    def reset_all(self):

        # Send
        for cpu in range(self.n_cpu):
            self.p_pipes[cpu].send(('reset', None))

        # Receive and normalize
        results = np.array([])
        for cpu in range(self.n_cpu):
            obs = self.p_pipes[cpu].recv()
            for i in range(len(obs)):
                if (obs[i] >= 0.0): obs[i] /= self.obs_max[i]
                if (obs[i] <  0.0): obs[i] /= self.obs_min[i]
            results = np.append(results, obs)

        return np.reshape(results, (-1,self.obs_dim))

    # Reset based on a done array
    def reset(self, done, obs_array):

        # Send
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                self.p_pipes[cpu].send(('reset', None))

        # Receive and normalize
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                obs = self.p_pipes[cpu].recv()
                for i in range(len(obs)):
                    if (obs[i] >= 0.0): obs[i] /= self.obs_max[i]
                    if (obs[i] <  0.0): obs[i] /= self.obs_min[i]
                obs_array[cpu] = obs

    # Get environment dimensions
    def get_dims(self):

        # Send
        self.p_pipes[0].send(('get_dims', None))

        # Receive
        results = np.array([])
        results = np.append(results, self.p_pipes[0].recv())

        return results

    # Get observations boundaries
    def get_obs_bounds(self):

        # Send
        self.p_pipes[0].send(('get_obs_bounds', None))

        # Receive
        obs_min, obs_max = self.p_pipes[0].recv()

        return obs_min, obs_max

    # Get action boundaries
    def get_act_bounds(self):

        # Send
        self.p_pipes[0].send(('get_act_bounds', None))

        # Receive
        act_min, act_max = self.p_pipes[0].recv()

        return act_min, act_max

    # Set cpu indices
    def set_cpus(self):

        # Send
        for cpu in range(self.n_cpu):
            self.p_pipes[cpu].send(('set_cpu', [cpu, self.n_cpu]))

    # Render environment
    def render(self, render):

        # Not all environments will render simultaneously
        # We use a list to store those that render and those that don't
        rnd = [[] for _ in range(self.n_cpu)]

        # Send
        for cpu in range(self.n_cpu):
            if (render[cpu]):
                self.p_pipes[cpu].send(('render', None))

        # Receive
        for cpu in range(self.n_cpu):
            if (render[cpu]):
                rnd[cpu] = self.p_pipes[cpu].recv()

        return rnd

    # Render environment
    def render_single(self, cpu):

        # Send
        self.p_pipes[cpu].send(('render', None))

        # Receive
        rgb = self.p_pipes[cpu].recv()

        return rgb

    # Close
    def close(self):

        # Close all envs
        for cpu in range(self.n_cpu):
            self.p_pipes[cpu].send(('close', None))
        for p in self.proc:
            p.terminate()
            p.join()

    # Take one step in all environments
    def step(self, actions):

        # Send
        for cpu in range(self.n_cpu):
            act = actions[cpu]
            for i in range(len(act)):
                if (act[i] >= 0.0): act[i] *= self.act_max[i]
                if (act[i] <  0.0): act[i] *= self.act_min[i]
            self.p_pipes[cpu].send(('step', act))

        # Receive
        nxt  = np.array([])
        rwd  = np.array([])
        done = np.array([], dtype=np.bool)
        for cpu in range(self.n_cpu):
            n, r, d = self.p_pipes[cpu].recv()
            nxt     = np.append(nxt, n)
            rwd     = np.append(rwd, r)
            done    = np.append(done, bool(d))

        nxt = np.reshape(nxt, (-1,self.obs_dim))

        return nxt, rwd, done

# Target function for process
def worker(env_name, name, pipe, p_pipe, path):
    env = gym.make(env_name)
    p_pipe.close()
    try:
        while True:
            # Receive command
            command, data = pipe.recv()

            # Execute command
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
            if (command == 'get_dims'):
                # Discrete or continuous action space
                if hasattr(env.action_space, "n"):
                    act_dim = env.action_space.n
                else:
                    act_dim = env.action_space.shape[0]
                obs_dim = env.observation_space.shape[0]
                pipe.send((act_dim, obs_dim))
            if (command == 'get_obs_bounds'):
                obs_min = env.observation_space.low
                obs_max = env.observation_space.high
                pipe.send((obs_min, obs_max))
            if (command == 'get_act_bounds'):
                act_min = env.action_space.low
                act_max = env.action_space.high
                pipe.send((act_min, act_max))
            if (command == 'set_cpu'):
                if hasattr(env, 'cpu'):
                    env.set_cpu(data[0], data[1])
            if command == 'close':
                pipe.send(None)
                break
    finally:
        env.close()
