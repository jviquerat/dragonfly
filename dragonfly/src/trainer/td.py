# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.core.constants  import *
from dragonfly.src.trainer.base    import *
from dragonfly.src.envs.par_envs   import *
from dragonfly.src.agent.agent     import *
from dragonfly.src.utils.timer     import *
from dragonfly.src.utils.report    import *
from dragonfly.src.utils.renderer  import *

###############################################
### Class for temporal-difference training
### env_pms   : environment parameters
### agent_pms : agent parameters
### path      : path for environment
### n_cpu     : nb of parallel environments
### n_ep_max  : max nb of episodes to unroll in a run
### pms       : parameters
class td(trainer_base):
    def __init__(self, env_pms, agent_pms, path, n_cpu, n_ep_max, pms):

        # Initialize environment
        self.env = par_envs(n_cpu, path, env_pms)

        # Initialize from input
        self.obs_dim      = self.env.obs_dim
        self.act_dim      = self.env.act_dim
        self.n_cpu        = n_cpu
        self.n_ep_max     = n_ep_max
        self.mem_size     = pms.mem_size
        self.n_stp_unroll = pms.n_stp_unroll*n_cpu
        self.btc_size     = pms.btc_size

        # Local variables
        self.unroll = 0

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.type,
                                          obs_dim = self.obs_dim,
                                          act_dim = self.act_dim,
                                          n_cpu   = self.n_cpu,
                                          size    = self.mem_size,
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

            # Loop over training steps
            while (not (self.unroll >= self.n_stp_unroll)):

                # Get actions
                act = self.agent.actions(obs)

                # Make one env step
                nxt, rwd, dne = self.env.step(act)

                # Store transition
                self.agent.store(obs, nxt, act, rwd, dne)

                # Update unrolling counter
                self.unroll += self.n_cpu

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

            # Train agent
            self.train()

            # Reset unroll
            self.unroll = 0

        # Last printing
        self.print_episode()

        # Write report data to file
        self.write_report(path, run)

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

        # Loop on unroll steps
        for i in range(self.n_stp_unroll):

            # Prepare training data
            self.agent.prepare_data(self.btc_size)
            self.agent.train(self.btc_size)

        self.timer_training.toc()
