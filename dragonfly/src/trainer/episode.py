# Custom imports
from dragonfly.src.trainer.base import *

###############################################
### Class for episode-based training
### env_pms     : environment parameters
### agent_pms   : agent parameters
### path        : path for environment
### n_steps_max : number of max steps
### pms         : parameters
class episode(base_trainer):
    def __init__(self, env_pms, agent_pms, path, n_stp_max, pms):

        # Initialize environment
        self.env = environments(path, env_pms)

        # Initialize from input
        self.obs_dim     = self.env.obs_dim
        self.act_dim     = self.env.act_dim
        self.n_stp_max   = n_stp_max
        self.n_ep_unroll = pms.n_ep_unroll*(mpi.size)
        self.n_ep_train  = pms.n_ep_train
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs
        self.size        = 1000*self.n_ep_train
        self.freq_report = 10
        self.update_type = "on_policy"

        # Optional modification of default args
        if hasattr(pms, "update"): self.update_type = pms.update

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
                                          n_cpu   = mpi.size,
                                          size    = self.size,
                                          pms     = agent_pms)

        # Initialize update
        self.update = update_factory.create(self.update_type)

        # Initialize counter
        self.counter = counter(mpi.size)

        # Initialize learning data report
        self.report = report(self.freq_report,
                             ["step", "episode", "score", "smooth_score"])

        # Initialize renderer
        self.rnd_style = "rgb_array"
        if hasattr(pms, "rnd_style"):
            self.rnd_style = pms.rnd_style
        self.renderer = renderer(mpi.size, self.rnd_style, pms.render_every)

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
                for cpu in range(mpi.size):
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
            self.update.update(self.agent,
                               size, btc_size, self.n_epochs)
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
