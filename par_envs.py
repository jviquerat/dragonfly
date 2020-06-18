# Generic imports
import gym
import time
import numpy           as np
import multiprocessing as mp

###############################################
### A wrapper class for parallel environments
class par_envs:
    def __init__(self, env_name, n_cpu):

        print(gym.__file__)

        # Init pipes and processes
        self.n_cpu        = n_cpu
        self.parent_pipes = []
        self.processes    = []

        # Start environments
        for cpu in range(n_cpu):
            parent_pipe, child_pipe = mp.Pipe()
            name                    = str(cpu)
            process = mp.Process(target = worker,
                                 name   = name,
                                 args   = (env_name, name, child_pipe))

            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)

            process.daemon = True
            process.start()

        # Handle action and observation dimensions
        act_dim, obs_dim = self.get_dims()
        self.act_dim = int(act_dim)
        self.obs_dim = int(obs_dim)

    # Reset all environments
    def reset(self):

        # Send
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))

        # Receive
        results = np.array([])
        for pipe in self.parent_pipes:
            results = np.append(results, pipe.recv())

        return np.reshape(results, (-1,self.obs_dim))

    # Reset a single environment
    def reset_single(self, cpu):

        # Send
        self.parent_pipes[cpu].send(('reset',None))

        # Receive
        results = np.array([])
        results = np.append(results, self.parent_pipes[cpu].recv())

        return np.reshape(results, (-1,self.obs_dim))

    # Get environment dimensions
    def get_dims(self):

        # Send
        self.parent_pipes[0].send(('get_dims',None))

        # Receive
        results = np.array([])
        results = np.append(results, self.parent_pipes[0].recv())

        return results

    # Render environment
    def render_single(self, cpu):

        # Send
        self.parent_pipes[cpu].send(('render', None))

        # Receive
        rgb = self.parent_pipes[cpu].recv()

        return rgb

    # Close
    def close(self):

        for p in self.processes:
            p.terminate()
            p.join()

    # Take one step in all environments
    def step(self, actions):

        # Send
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', action))

        # Receive
        nxt  = np.array([])
        rwd  = np.array([])
        done = np.array([], dtype=np.bool)
        for pipe in self.parent_pipes:
            n, r, d = pipe.recv()
            nxt     = np.append(nxt, n)
            rwd     = np.append(rwd, r)
            done    = np.append(done, bool(d))

        nxt = np.reshape(nxt, (-1,self.obs_dim))

        return nxt, rwd, done

# Target function for process
def worker(env_name, name, pipe):
    env = gym.make(env_name)
    #env = gym.wrappers.Monitor(env,
    #                           './vids/'+str(time.time())+'/',
    #                           video_callable=None)
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
                act_dim = env.action_space.n
                obs_dim = env.observation_space.shape[0]
                pipe.send((act_dim, obs_dim))
            if command == 'close':
                pipe.send(None)
                break
    finally:
        env.close()
