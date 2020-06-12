# Generic imports
import gym
import multiprocessing as mp

###############################################
### A wrapper class for parallel environments
class par_envs:
    def __init__(self, env_name, n_cpu):

        self.n_cpu = n_cpu
        self.parent_pipes, self.processes = [], []

        for cpu in range(n_cpu):
            parent_pipe, child_pipe = mp.Pipe()
            process = mp.Process(target=worker,
                                 name=str(cpu),
                                 args=(env_name, child_pipe)
                    #parent_pipe, _obs_buffer, self.error_queue))

            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)

            #process.daemon = daemon
            process.start()
            child_pipe.close()

    def reset():

    def step():

# Target function for process
def worker(env_name, pipe):#, parent_pipe):
    env = gym.make(env_name)
    #parent_pipe.close()
    #try:
    while True:
        # Receive command
        command, data = pipe.recv()

        # Execute command
        if command == 'reset':
            obs = env.reset()
            pipe.send((observation, True))
        if command == 'step':
            nxt, rwd, done, _ = env.step(data)
            #if done:
            #    observation = env.reset()
            pipe.send(((nxt, rwd, done), True))
        if command == 'seed':
            env.seed(data)
            pipe.send((None, True))
        if command == 'close':
            pipe.send((None, True))
            break
    #except (KeyboardInterrupt, Exception):
    #    error_queue.put((index,) + sys.exc_info()[:2])
    #    pipe.send((None, False))
    #finally:
    #    env.close()
