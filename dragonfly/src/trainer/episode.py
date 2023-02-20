# Custom imports
from dragonfly.src.trainer.base import *

###############################################
### Class for episode-based training
### env_pms   : environment parameters
### agent_pms : agent parameters
### path      : path for environment
### n_cpu     : nb of parallel environments
### pms       : parameters
class episode(trainer_base):
    def __init__(self, env_pms, agent_pms, path, n_cpu, n_stp_max, pms):

        # Initialize environment
        self.env = par_envs(n_cpu, path, env_pms)

        # Initialize from input
        self.obs_dim     = self.env.obs_dim
        self.act_dim     = self.env.act_dim
        self.n_cpu       = n_cpu
        self.n_stp_max   = n_stp_max
        self.n_ep_unroll = pms.n_ep_unroll*n_cpu
        self.n_ep_train  = pms.n_ep_train
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs
        self.size        = 1000*self.n_ep_train
        self.freq_report = 10

        # Optional monitoring
        self.monitoring = False
        if hasattr(pms, "monitoring"):  self.monitoring = pms.monitoring

        # Local variables
        self.lengths = np.array([], dtype=int)
        self.unroll  = 0

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.type,
                                          obs_dim = self.obs_dim,
                                          act_dim = self.act_dim,
                                          n_cpu   = self.n_cpu,
                                          size    = self.size,
                                          pms     = agent_pms)

        # Initialize counter
        self.counter = counter(self.n_cpu)

        # Initialize learning data report
        self.report = report(self.freq_report,
                             ["step", "episode", "score", "smooth_score"])

        # Initialize renderer
        self.rnd_style = "rgb_array"
        if hasattr(pms, "rnd_style"):
            self.rnd_style = pms.rnd_style
        self.renderer = renderer(self.n_cpu, self.rnd_style, pms.render_every)

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_training = timer("training ")

    # Loop
    def loop(self, path, run):

        # Start global timer
        self.timer_global.tic()

        # Reset environment
        obs = self.env.reset_all()

        # Reset counter
        self.counter.reset()

        # Loop until max episode number is reached
        while (self.counter.step < self.n_stp_max):

            # Prepare inner training loop
            self.agent.pre_loop()

            # Loop over training episodes
            while (not (self.unroll >= self.n_ep_unroll)):

                # Get actions
                act = self.agent.actions(obs)

                # Make one env step
                nxt, rwd, dne, trc = self.env.step(act)

                # Store transition
                self.agent.store(obs, nxt, act, rwd, dne, trc)
                self.monitor(path, run, obs, act)

                # Update counter
                self.counter.update(rwd)

                # Handle rendering
                self.renderer.store(self.env)

                # Finish if some episodes are done
                for cpu in range(self.n_cpu):
                    if (dne[cpu]):
                        self.lengths = np.append(self.lengths,
                                                 self.counter.ep_step[cpu])
                        self.store_report(cpu)
                        self.print_episode()
                        self.renderer.finish(path, run, self.counter.ep, cpu)
                        best = self.counter.reset_ep(cpu)
                        name = path+"/"+str(run)+"/"+self.agent.name
                        if best: self.agent.save(name)
                        self.unroll += 1

                # Update observation
                obs = nxt

                # Reset only finished environments
                self.env.reset(dne, obs)

            # Finalize inner training loop
            self.agent.post_loop()

            # Write report data to file
            self.write_report(path, run)

            # Train agent
            self.timer_training.tic()
            size     = np.sum(self.lengths[-self.n_ep_train:])
            btc_size = math.floor(size*self.btc_frac)
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

            # Reset unroll
            self.unroll = 0

        # Last printing
        self.print_episode()

        # Last writing
        self.write_report(path, run, force=True)

        # Close timers and show
        self.timer_global.toc()
        self.timer_global.show()
        self.env.timer_env.show()
        self.agent.timer_actions.show()
        self.timer_training.show()
