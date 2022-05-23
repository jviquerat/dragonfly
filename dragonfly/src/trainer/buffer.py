# Generic imports
import math
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

###############################################
### Class for buffer-based training
### obs_dim  : dimension of observations
### act_dim  : dimension of actions
### pol_dim  : true dimension of the actions provided to the env
### n_cpu    : nb of parallel environments
### n_ep_max : max nb of episodes to unroll in a run
### pms      : parameters
class buffer(trainer_base):
    def __init__(self, obs_dim, act_dim,
                 pol_dim, n_cpu, n_ep_max, pms):

        # Initialize from input
        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self.pol_dim   = pol_dim
        self.n_cpu     = n_cpu
        self.n_ep_max  = n_ep_max
        self.buff_size = pms.buff_size
        self.n_buff    = pms.n_buff
        self.btc_frac  = pms.batch_frac
        self.n_epochs  = pms.n_epochs

        # pol_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.buff = buff(self.n_cpu,
                        ["obs", "nxt", "act", "lgp", "rwd", "dne", "stp", "trm", "bts"],
                        [obs_dim, obs_dim, pol_dim, 1, 1, 1, 1, 1, 1])
        self.gbuff = gbuff(self.n_buff*self.buff_size,
                           ["obs", "act", "adv", "tgt", "lgp"],
                           [obs_dim, pol_dim, 1, 1, 1])

        # Initialize learning data report
        self.report = report(["episode", "step",
                              "score",   "smooth_score",
                              "length",  "smooth_length",
                              "entropy", "smooth_entropy"])

        # Initialize renderer
        self.renderer = renderer(self.n_cpu, pms.render_every)

        # Initialize counter
        self.counter = counter(self.n_cpu, self.n_ep_max)

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
        while (not (self.counter.ep >= self.n_ep_max)):

            # Reset local buffer
            self.buff.reset()

            # Loop over buff size
            while (not (self.buff.size() >= self.buff_size)):

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
                entropy = agent.entropy(obs)
                self.counter.update(rwd, entropy)

                # Handle rendering
                self.renderer.store(env.render(self.renderer.render), env.rnd_style)

                # Finish if some episodes are done
                self.finish_episodes(agent, path, run, dne)

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
    def finish_episodes(self, agent, path, run, done):

        # Loop over environments and finalize/reset
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                self.store_report(self.counter, self.report, cpu)
                self.print_episode(self.counter, self.report)
                self.renderer.finish(path, self.counter.ep, cpu)
                best = self.counter.reset_ep(cpu)
                if best: agent.save(path+"/"+str(run)+"/"+agent.name)

    # Train
    def train(self, agent):

        # Compute training buff size and batch size
        size      = self.n_buff*self.buff_size
        btc_size  = math.floor(size*self.btc_frac)

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
