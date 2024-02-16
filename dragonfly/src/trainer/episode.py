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
        super().__init__(self, env_pms=env_pms, path=path, n_stp_max=n_stp_max)

        self.n_ep_unroll = pms.n_ep_unroll*(mpi.size)
        self.n_ep_train  = pms.n_ep_train
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs
        self.size        = 1000*self.n_ep_train
        self.freq_report = 10
        self.update_type = "on_policy"

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

        # Initialize learning data report
        self.report = report(self.freq_report,
                             ["step", "episode", "score", "smooth_score"])

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

                nxt, _, dne, _ = self.apply_next_step(obs,path,run)

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

        self.end_training(path, run)
