# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.core.constants        import *
from dragonfly.src.trainer.base          import *
from dragonfly.src.envs.par_envs         import *
from dragonfly.src.agent.agent           import *
from dragonfly.src.utils.timer           import *
from dragonfly.src.utils.report          import *
from dragonfly.src.utils.renderer        import *

###############################################
### Class for episode-based training
### env_pms   : environment parameters
### agent_pms : agent parameters
### path      : path for environment
### n_cpu     : nb of parallel environments
### n_ep_max  : max nb of episodes to unroll in a run
### pms       : parameters
class episode(trainer_base):
    def __init__(self, env_pms, agent_pms, path, n_cpu, n_ep_max, pms):

        # Initialize environment
        self.env = par_envs(n_cpu, path, env_pms)

        # Initialize from input
        self.obs_dim     = self.env.obs_dim
        self.act_dim     = self.env.act_dim
        self.n_cpu       = n_cpu
        self.n_ep_max    = n_ep_max
        self.n_ep_unroll = pms.n_ep_unroll*n_cpu
        self.n_ep_train  = pms.n_ep_train
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs
        self.size        = 1000*self.n_ep_train

        # Local variables
        self.unroll = 0

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.type,
                                          obs_dim = self.obs_dim,
                                          act_dim = self.act_dim,
                                          n_cpu   = self.n_cpu,
                                          size    = self.size,
                                          pms     = agent_pms)

        # Initialize learning data report
        self.report = report(["episode", "step",
                              "score",  "smooth_score",
                              "length", "smooth_length"])

        # Initialize renderer
        self.renderer = renderer(self.n_cpu,
                                 self.env.rnd_style,
                                 pms.render_every)

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_training = timer("training ")

    # Loop
    def loop(self, path, run):

        # Start global timer
        self.timer_global.tic()

        # Reset environment
        obs = self.env.reset_all()

        # Loop until max episode number is reached
        while (not (self.agent.counter.ep >= self.n_ep_max)):

            # Prepare inner training loop
            self.agent.pre_loop()

            # Loop over training episodes
            while (not (self.unroll >= self.n_ep_unroll)):

                # Get actions
                act = self.agent.actions(obs)

                # Make one env step
                nxt, rwd, dne = self.env.step(act)

                # Store transition
                self.agent.store(obs, nxt, act, rwd, dne)

                # Handle rendering
                rnd = self.env.render(self.renderer.render)
                self.renderer.store(rnd)

                # Finish if some episodes are done
                self.finish_episodes(path, run, dne)

                # Update observation
                obs = nxt

                # Reset only finished environments
                self.env.reset(dne, obs)

            # Finalize inner training loop
            self.agent.post_loop()

            # Write report data to file
            self.write_report(path, run)

            # Train agent
            self.train()

            # Reset unroll
            self.unroll = 0

        # Last printing
        self.print_episode()

        # Close timers and show
        self.timer_global.toc()
        self.timer_global.show()
        self.env.timer_env.show()
        self.timer_training.show()

    # Finish if some episodes are done
    def finish_episodes(self, path, run, done):

        # Loop over environments and finalize/reset
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                self.agent.counter.update_global_step(cpu)
                self.store_report(cpu)
                self.print_episode()
                self.renderer.finish(path, self.agent.counter.ep, cpu)
                best = self.agent.counter.reset_ep(cpu)
                name = path+"/"+str(run)+"/"+self.agent.name
                if best: self.agent.save(name)
                self.unroll += 1

    # Train
    def train(self):

        self.timer_training.tic()

        # Retrieve size of training buffer using stored episode lengths
        lengths  = self.report.get("length")
        size     = np.sum(lengths[-self.n_ep_train:])
        btc_size = math.floor(size*self.btc_frac)

        # Loop on epochs
        for epoch in range(self.n_epochs):

            # Prepare training data
            lgt = self.agent.prepare_data(size)

            # Visit all available history
            done = False
            btc  = 0
            while not done:
                start = btc*btc_size
                end   = min((btc+1)*btc_size, lgt)

                self.agent.train(start, end)

                btc += 1
                if (end == lgt): done = True

        self.timer_training.toc()
