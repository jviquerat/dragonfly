# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.core.constants        import *
from dragonfly.src.trainer.base          import *
from dragonfly.src.terminator.terminator import *
from dragonfly.src.utils.timer           import *
from dragonfly.src.utils.buff            import *
from dragonfly.src.utils.report          import *
from dragonfly.src.utils.renderer        import *
from dragonfly.src.utils.counter         import *
from dragonfly.src.utils.error           import *

###############################################
### Class for temporal-difference training
### pms : parameters
class td(trainer_base):
    def __init__(self, obs_dim, act_dim,
                 pol_dim, n_cpu, n_ep_max, pms):

        # Initialize from input
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.pol_dim  = pol_dim
        self.n_cpu    = n_cpu
        self.n_ep_max = n_ep_max
        self.mem_size = pms.mem_size
        self.btc_size = pms.btc_size

        # Check that n_cpu is 1
        if (n_cpu != 1):
            error("td",
                  "init",
                  "td learning does not support parallel envs")

        # pol_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.buff = buff(self.n_cpu,
                        ["obs", "nxt", "act", "rwd", "dne", "stp", "trm"],
                        [obs_dim, obs_dim, pol_dim, 1, 1, 1, 1])
        self.gbuff = gbuff(self.mem_size,
                           ["obs", "act", "tgt"],
                           [obs_dim, pol_dim, 1])

        # Initialize learning data report
        self.report = report(["episode",
                              "score",  "smooth_score",
                              "length", "smooth_length",
                              "step"])

        # Initialize renderer
        self.renderer = renderer(self.n_cpu, pms.render_every)

        # Initialize counter
        self.counter = counter(self.n_cpu,
                               self.n_ep_max,
                               "td",
                               n_stp_unroll=1)

        # Initialize terminator
        self.terminator = terminator_factory.create(pms.terminator.type,
                                                    n_cpu = self.n_cpu,
                                                    pms   = pms.terminator)

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_env      = timer("env      ")
        self.timer_actions  = timer("actions  ")
        self.timer_training = timer("training ")

    # Loop
    def loop(self, path, run, env, agent):

        # Start global timer
        self.timer_global.tic()

        # Reset environment
        obs = env.reset_all()

        # Loop until max episode number is reached
        while (not self.counter.done_max_ep()):

            # Reset local buffer
            self.buff.reset()

            # Loop over training steps
            while (not self.counter.done_stp_unroll()):

                # Get actions
                self.timer_actions.tic()
                act = agent.get_actions(obs)
                self.timer_actions.toc()

                # Make one env step
                self.timer_env.tic()
                nxt, rwd, dne = env.step(act)
                self.timer_env.toc()

                # Store transition
                stp = self.counter.ep_step
                self.buff.store(["obs", "nxt", "act", "rwd", "dne", "stp"],
                                [ obs,   nxt,   act,   rwd,   dne,   stp ])

                # Update counter
                self.counter.update(rwd)
                self.counter.unroll += 1

                # Handle rendering
                self.renderer.store(env.render(self.renderer.render))

                # Finish if some episodes are done
                self.finish_episodes(path, dne)

                # Update observation
                obs = nxt

                # Reset only finished environments
                self.timer_env.tic()
                env.reset(dne, obs)
                self.timer_env.toc()

            # Finalize buffers for training
            self.terminator.terminate(self.buff)
            names = ["obs", "nxt", "act", "rwd", "trm"]
            data  = self.buff.serialize(names)
            gobs, gnxt, gact, grwd, gtrm = (data[name] for name in names)
            gtgt = agent.compute_target(gobs, gnxt, gact, grwd, gtrm)


            # Store in global buffers
            self.gbuff.store(["obs", "tgt", "act"],
                             [gobs,  gtgt,  gact ])

            # Train agent
            self.timer_training.tic()
            self.train(agent)
            self.timer_training.toc()

        # Last printing
        self.print_episode(self.counter, self.report)

        # Write report data to file
        self.write_report(agent, self.report, path, run)

        # Close timers and show
        self.timer_global.toc()
        self.timer_global.show()
        self.timer_env.show()
        self.timer_actions.show()
        self.timer_training.show()

    # Finish if some episodes are done
    def finish_episodes(self, path, done):

        # Loop over environments and finalize/reset
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                self.store_report(self.counter, self.report, cpu)
                self.print_episode(self.counter, self.report)
                self.renderer.finish(path, self.counter.ep, cpu)
                self.counter.reset_ep(cpu)

    # Train
    def train(self, agent):

        # Retrieve data
        names = ["obs", "act", "tgt"]
        data  = self.gbuff.get_buffers(names, self.btc_size)
        obs, act, tgt = (data[name] for name in names)

        if (len(obs) < self.btc_size): return

        agent.train(obs, act, tgt, self.btc_size)
