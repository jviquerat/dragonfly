# Generic imports
import math
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
### Class for buffer-based training
### env_pms   : environment parameters
### agent_pms : agent parameters
### path      : path for environment
### pms       : parameters
class buffer(trainer_base):
    def __init__(self, env_pms, agent_pms, path, n_cpu, n_stp_max, pms):

        # Initialize environment
        self.env   = par_envs(n_cpu, path, env_pms)

        # Initialize from input
        self.obs_dim   = self.env.obs_dim
        self.act_dim   = self.env.act_dim
        self.n_cpu     = n_cpu
        self.n_stp_max = n_stp_max
        self.buff_size = pms.buff_size
        self.n_buff    = pms.n_buff
        self.btc_frac  = pms.batch_frac
        self.n_epochs  = pms.n_epochs
        self.size      = self.n_buff*self.buff_size

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.type,
                                          obs_dim = self.obs_dim,
                                          act_dim = self.act_dim,
                                          n_cpu   = self.n_cpu,
                                          size    = self.size,
                                          pms     = agent_pms)

        # Initialize learning data report
        self.report = report(["episode", "step",
                              "score",   "smooth_score"])

        # Initialize renderer
        self.renderer = renderer(self.n_cpu,
                                 "rgb_array",
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
        while (self.agent.counter.step < self.n_stp_max):

            # Prepare inner training loop
            self.agent.pre_loop()

            # Loop over buff size
            while (not (self.agent.buff.size() >= self.buff_size)):

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
            self.agent.post_loop(style="buffer")

            # Write report data to file
            self.write_report(path, run)

            # Train agent
            self.train()

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
                self.store_report(cpu)
                self.print_episode()
                self.renderer.finish(path, self.agent.counter.ep, cpu)
                best = self.agent.counter.reset_ep(cpu)
                name = path+"/"+str(run)+"/"+self.agent.name
                if best: self.agent.save(name)

    # Train
    def train(self):

        self.timer_training.tic()

        # Train policy and v_value
        btc_size = math.floor(self.size*self.btc_frac)
        for epoch in range(self.n_epochs):

            # Prepare training data
            lgt = self.agent.prepare_data(self.size)

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
