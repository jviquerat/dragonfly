# Custom imports
from dragonfly.src.trainer.base import *

###############################################
### Class for temporal-difference training
### env_pms     : environment parameters
### agent_pms   : agent parameters
### path        : path for environment
### n_steps_max : number of max steps
### pms         : parameters
class td(base_trainer):
    def __init__(self, env_pms, agent_pms, path, n_stp_max, pms):
        super().__init__(self, env_pms=env_pms, path=path, n_stp_max=n_stp_max)
        
        self.mem_size     = pms.mem_size
        self.n_stp_unroll = pms.n_stp_unroll*mpi.size
        self.btc_size     = pms.btc_size
        self.freq_report  = max(int(self.n_stp_max/(freq_report*self.n_stp_unroll)),1)
        self.update_type = "off_policy"

        # Optional modification of default args
        if hasattr(pms, "update"): self.update_type = pms.update

        # Optional parameters
        self.monitoring = False
        if hasattr(pms, "monitoring"):  self.monitoring = pms.monitoring

        # Local variables
        self.unroll = 0

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.type,
                                          obs_dim = self.obs_dim,
                                          act_dim = self.act_dim,
                                          n_cpu   = mpi.size,
                                          size    = self.mem_size,
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

            # Loop over training steps
            while (not (self.unroll >= self.n_stp_unroll)):

                # Get actions
                act = self.agent.actions(obs)

                # Make one env step
                nxt, rwd, dne, trc = self.env.step(act)

                # Store transition
                self.agent.store(obs, nxt, act, rwd, dne, trc)
                self.monitor(path, run, obs, act)

                # Update counter
                self.counter.update(rwd)

                # Update unrolling counter
                self.unroll += mpi.size

                # Handle rendering
                self.renderer.store(self.env)

                # Finish if some episodes are done
                for cpu in range(mpi.size):
                    if (dne[cpu]):
                        self.store_report(cpu)
                        self.print_episode()
                        self.renderer.finish(path, run, self.counter.ep, cpu)
                        best = self.counter.reset_ep(cpu)
                        name = path+"/"+str(run)+"/"+self.agent.name
                        if best: self.agent.save(name)

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
            self.update.update(self.agent, self.btc_size, self.n_stp_unroll)
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
