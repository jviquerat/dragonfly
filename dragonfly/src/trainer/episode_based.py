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
class episode_based(trainer_base):
    def __init__(self, obs_dim, act_dim,
                 pol_act_dim, n_cpu, n_ep_max, pms):

        # Initialize from input
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.pol_act_dim = pol_act_dim
        self.n_cpu       = n_cpu
        self.n_ep_max    = n_ep_max
        self.n_ep_unroll = pms.n_ep_unroll
        self.n_ep_train  = pms.n_ep_train
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs

        # Check that n_ep_unroll is a multiple of n_cpu
        #if (n_ep_unroll%n_cpu != 0):
            #error("episode_based",
            #      "init",
            #      "n_ep_unroll is not a multiple of n_cpu")
        if (n_cpu != 1):
            error("episode_based",
                  "init",
                  "episode-based learning does not support parallel envs")

        # pol_act_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.loc_buff = loc_buff(self.n_cpu,
                                 self.obs_dim,
                                 self.pol_act_dim)
        self.glb_buff = glb_buff(self.n_cpu,
                                 self.obs_dim,
                                 self.pol_act_dim)

        # Initialize learning data report
        self.report_fields = ["episode",
                              "score",  "smooth_score",
                              "length", "smooth_length",
                              "step"]
        self.report = report(self.report_fields)

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
            self.loc_buff.reset()

            # Loop over training episodes
            while (not self.counter.done_unroll()):

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
                self.loc_buff.store(obs, nxt, act, rwd, dne, stp)

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
            self.terminator.terminate(self.loc_buff)
            obs, nxt, act, rwd, trm, bts = self.loc_buff.serialize()
            tgt, adv = agent.compute_returns(obs, nxt, act, rwd, trm, bts)

            # Store in global buffers
            self.glb_buff.store(obs, adv, tgt, act)

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

        # Save previous policy
        agent.policy.save_prv()

        # Retrieve size of training buffer using stored episode lengths
        lengths  = self.report.get("length")
        size     = np.sum(lengths[-self.n_ep_train:])
        btc_size = math.floor(size*self.btc_frac)

        # Train policy and v_value
        for epoch in range(self.n_epochs):

            # Retrieve data
            obs, act, adv, tgt = self.glb_buff.get_buffers(size)

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

                agent.train(btc_obs, btc_act, btc_adv, btc_tgt, end-start)

                btc += 1
                if (end == lgt): done = True
