# Generic imports
import sys
import gymnasium as gym

# Custom imports
from dragonfly.src.env.mpi import mpi
from dragonfly.src.env.utils import find_class_in_folder, import_class_from_file
from dragonfly.src.utils.error import error

###############################################
# Worker class for slave processes
class worker():
    def __init__(self,
                 env_name,
                 cpu,
                 path_hint,
                 args):

        # Build environment
        try: # Test if this is a gym environment
            if args is not None:
                self.env = gym.make(env_name,
                                    render_mode="rgb_array",
                                    **args.__dict__)
            else:
                self.env = gym.make(env_name,
                                    render_mode="rgb_array")
        except: # Othwerise, look for env_name class in all files of path_hint folder
            files = find_class_in_folder(path_hint, env_name)

            if len(files) > 1:
                error("worker", "init",
                      f"Found more than one file containing class: {env_name}")

            builder = import_class_from_file(files[0], env_name)
            try:
                if args is not None:
                    self.env = builder(cpu, **args.__dict__)
                else:
                    self.env = builder(cpu)
            except:
                if args is not None:
                    self.env = builder(**args.__dict__)
                else:
                    self.env = builder()

    # Working function for slaves
    def work(self):
        while True:
            data    = None
            data    = mpi.comm.scatter(data, root=0)
            command = data[0]
            data    = data[1]

            # Execute commands
            if command == 'step':
                nxt, rwd, done, trunc = self.step(data)
                mpi.comm.gather((nxt, rwd, done, trunc), root=0)

            if command == 'reset':
                obs = self.reset(data)
                mpi.comm.gather((obs), root=0)

            if command == 'render':
                rnd = self.render(data)
                mpi.comm.gather((rnd), root=0)

            if command == 'close':
                self.close()
                mpi.finalize()
                break

    # Stepping
    def step(self, data):
        nxt, rwd, done, trunc, _ = self.env.step(data)
        if ((not done) and trunc): done = True

        return nxt, rwd, done, trunc

    # Resetting
    def reset(self, data):
        if data: obs, _ = self.env.reset()
        else: obs = None

        return obs

    # Rendering
    def render(self, data):
        rnd = None
        if (data): rnd = self.env.render()

        return rnd

    # Closing
    def close(self):
        self.env.close()
