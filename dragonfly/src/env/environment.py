# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.env.worker  import worker
from dragonfly.src.env.mpi     import mpi
from dragonfly.src.env.spaces  import environment_spaces
from dragonfly.src.utils.timer import timer

###############################################
### A wrapper class for parallel environments
class environment:
    # Class variable
    def __init__(self, path, pms):

        # Default values
        self.name = pms.name
        self.args = None

        # Optional values in parameters
        if hasattr(pms, "args"): self.args = pms.args

        # Generate workers
        self.worker = worker(self.name, mpi.rank, path, self.args)

        # Set all slaves to wait for instructions
        if (mpi.rank != 0): self.worker.work()

        # Declare spaces object
        self.spaces = environment_spaces(pms, self.get_spaces())

        # Output dimensions
        self.spaces.print()

        # Initialize an observation array for stacking
        self.nxt = np.zeros((mpi.size,
                             self.spaces.processed_obs_dim(),
                             self.spaces.obs_stack()))

        # Initialize timer
        self.timer_env = timer("env      ")

    # Take one step in all environments
    def step(self, actions):

        self.timer_env.tic()

        rwd   = np.zeros((mpi.size))
        done  = np.zeros((mpi.size))
        trunc = np.zeros((mpi.size))

        # The whole step part is looped over frameskip times
        # Yet if a stack of observations is used, we keep the
        # fine-grain history of observations
        for _ in range(self.spaces.obs_frameskip()):

            # Send
            data = [('step', None)]*mpi.size
            for p in range(mpi.size):
                act     = self.spaces.process_actions(actions[p])
                data[p] = ('step', act)
            mpi.comm.scatter(data, root=0)

            # Main process executing
            n, r, d, t = self.worker.step(data[0][1])

            # Handle stacked observations
            # Latest observation is put in last position
            for p in range(mpi.size):
                for s in range(self.spaces.obs_stack()-1):
                    self.nxt[p,:,s] = self.nxt[p,:,s+1]

            # Receive
            data = mpi.comm.gather((n, r, d, t), root=0)

            for p in range(mpi.size):
                vals       = data[p]
                n, r, d, t = vals[0], vals[1], vals[2], vals[3]
                nn         = self.spaces.process_observations(n)

                # New observation is put in last position
                self.nxt[p,:,-1] = nn[:].flatten()

                # Reward is updated to account for possible frameskip
                rwd  [p]        += r
                done [p]         = bool(d)
                trunc[p]         = bool(t)

        # Reshape observations for returning
        nxt = np.reshape(self.nxt, (mpi.size, self.spaces.true_obs_dim())).copy()

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
        data = mpi.comm.gather((obs), root=0)
        for p in range(mpi.size):
            obs = self.spaces.process_observations(data[p])
            self.nxt[p,:,:]  = 0.0
            self.nxt[p,:,-1] = obs[:].flatten()

        nxt = np.reshape(self.nxt, (mpi.size, self.spaces.true_obs_dim())).copy()

        return nxt

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
                obs = self.spaces.process_observations(data[p])
                obs_array[p,:] = np.tile(obs.flatten(), self.spaces.obs_stack())[:]

                self.nxt[p,:,:]  = 0.0
                self.nxt[p,:,-1] = obs[:].flatten()

        return obs_array

    # Get environment spaces
    def get_spaces(self):

        return [self.worker.env.action_space, self.worker.env.observation_space]

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
