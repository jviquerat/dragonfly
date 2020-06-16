# Generic imports
import gym
import multiprocessing as mp

###############################################
### A wrapper class for parallel environments
class par_envs:
    def __init__(self, env_name, n_cpu):

        self.n_cpu        = n_cpu
        self.parent_pipes = []
        self.processes    = []

        for cpu in range(n_cpu):
            parent_pipe, child_pipe = mp.Pipe()
            name                    = str(cpu)
            process = mp.Process(target = worker,
                                 name   = name,
                                 args   = (env_name, name, child_pipe))
                    #parent_pipe, _obs_buffer, self.error_queue))

            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)

            process.daemon = True
            process.start()
            #child_pipe.close()

        act_dim, obs_dim = self.get_dims()
        self.act_dim = act_dim
        self.obs_dim = obs_dim

    def reset(self):

        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        result = zip(*[pipe.recv() for pipe in self.parent_pipes])

        return results

    def reset_single(self, idx):

        self.parent_pipes[idx].send(('reset',None))
        results = self.parent_pipes[idx].recv()

        return results

    def get_dims(self):

        self.parent_pipes[0].send(('get_dims',None))
        results = self.parent_pipes[0].recv()

        return results

    def step(self, actions):

        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', action))
        results = zip(*[pipe.recv() for pipe in self.parent_pipes])

        return results

# Target function for process
def worker(env_name, name, pipe):#, parent_pipe):
    env = gym.make(env_name)
    print('Started env #'+name)
    #parent_pipe.close()
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
            if (command == 'get_dims'):
                act_dim = env.action_space.n
                obs_dim = env.observation_space.shape[0]
                pipe.send((act_dim, obs_dim))
            if command == 'close':
                pipe.send(None)
                break
    finally:
        env.close()
    #except (KeyboardInterrupt, Exception):
    #    error_queue.put((index,) + sys.exc_info()[:2])
    #    pipe.send((None, False))
