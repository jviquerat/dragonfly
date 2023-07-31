# Custom imports
from dragonfly.src.trainer.base import *

###############################################
### Class for buffer-based training
### env_pms     : environment parameters
### agent_pms   : agent parameters
### path        : path for environment
### n_steps_max : number of max steps
### pms         : parameters
class buffer(base_trainer):
    def __init__(self, env_pms, agent_pms, path, n_stp_max, pms):

        # Initialize environment
        self.env = environments(path, env_pms)

        # Initialize from input
        self.obs_dim     = self.env.obs_dim
        self.act_dim     = self.env.act_dim
        self.n_stp_max   = n_stp_max
        self.buff_size   = pms.buff_size
        self.n_buff      = pms.n_buff
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs
        self.size        = self.n_buff*self.buff_size
        self.freq_report = max(int(n_stp_max/(freq_report*self.buff_size)),1)
        self.update_type = "online"

        # Optional modification of default args
        if hasattr(pms, "update"): self.update_type = pms.update

        # Optional monitoring
        self.monitoring = False
        if hasattr(pms, "monitoring"): self.monitoring = pms.monitoring

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

            # Loop over buff size
            while (not (self.agent.buff.size() >= self.buff_size)):

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
            self.agent.post_loop(style="buffer")

            # Write report data to file
            self.write_report(path, run)

            # Train agent
            self.timer_training.tic()
            btc_size = math.floor(self.size*self.btc_frac)
            self.update.update(self.agent,
                               self.size, btc_size, self.n_epochs)
            self.timer_training.toc()

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
