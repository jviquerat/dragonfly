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
### Class for episode-based training
### pms : parameters
class episode(trainer_base):
    def __init__(self, obs_dim, act_dim,
                 pol_dim, n_cpu, n_ep_max, pms):

        # Initialize from input
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.pol_dim     = pol_dim
        self.n_cpu       = n_cpu
        self.n_ep_max    = n_ep_max
        self.n_ep_unroll = pms.n_ep_unroll
        self.n_ep_train  = pms.n_ep_train
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs

        # Check that n_ep_unroll is a multiple of n_cpu
        if (n_cpu != 1):
            error("episode",
                  "init",
                  "episode-based learning does not support parallel envs")

        # pol_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.buff = buff(self.n_cpu,
                        ["obs", "nxt", "act", "lgp", "rwd", "dne", "stp", "trm", "bts"],
                        [obs_dim, obs_dim, pol_dim, 1, 1, 1, 1, 1, 1])
        self.gbuff = gbuff(1000*self.n_ep_train,
                           ["obs", "act", "adv", "tgt", "lgp"],
                           [obs_dim, pol_dim, 1, 1, 1])

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
                               "episode",
                               n_ep_unroll=self.n_ep_unroll)

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

            # Loop over training episodes
            while (not self.counter.done_ep_unroll()):

                # Get actions
                self.timer_actions.tic()
                act, lgp = agent.get_actions(obs)
                self.timer_actions.toc()

                # Make one env step
                self.timer_env.tic()
                nxt, rwd, dne = env.step(act)
                self.timer_env.toc()

                # Store transition
                stp = self.counter.ep_step
                self.buff.store(["obs", "nxt", "act", "lgp", "rwd", "dne", "stp"],
                                [ obs,   nxt,   act,   lgp,   rwd,   dne,   stp ])

                # Update counter
                self.counter.update(rwd)

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
            names = ["obs", "nxt", "act", "lgp", "rwd", "trm", "bts"]
            data  = self.buff.serialize(names)
            gobs, gnxt, gact, glgp, grwd, gtrm, gbts = (data[name] for name in names)
            gtgt, gadv = agent.compute_returns(gobs, gnxt, gact, grwd, gtrm, gbts)

            # Store in global buffers
            self.gbuff.store(["obs", "adv", "tgt", "act", "lgp"],
                             [gobs,  gadv,  gtgt,  gact,  glgp ])

            # Write report data to file
            self.write_report(agent, self.report, path, run)

            # Train agent
            self.timer_training.tic()
            self.train(agent)
            self.timer_training.toc()

        # Last printing
        self.print_episode(self.counter, self.report)

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
                self.counter.unroll += 1

    # Train
    def train(self, agent):

        # Retrieve size of training buffer using stored episode lengths
        lengths  = self.report.get("length")
        size     = np.sum(lengths[-self.n_ep_train:])
        btc_size = math.floor(size*self.btc_frac)

        # Train policy and v_value
        for epoch in range(self.n_epochs):

            # Retrieve data
            names = ["obs", "adv", "tgt", "act", "lgp"]
            data  = self.gbuff.get_buffers(names, size)
            obs, adv, tgt, act, lgp = (data[name] for name in names)

            # Visit all available history
            done = False
            btc  = 0
            while not done:
                lgt   = len(obs)
                start = btc*btc_size
                end   = min((btc+1)*btc_size, lgt)

                btc_obs = obs[start:end]
                btc_act = act[start:end]
                btc_adv = adv[start:end]
                btc_tgt = tgt[start:end]
                btc_lgp = lgp[start:end]

                agent.train(btc_obs, btc_act, btc_adv,
                            btc_tgt, btc_lgp, end-start)

                btc += 1
                if (end == lgt): done = True
