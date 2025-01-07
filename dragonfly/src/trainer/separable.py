# Custom imports
from dragonfly.src.trainer.base import *

###############################################
### Separable buffer-based trainer class
class separable(base_trainer):
    def __init__(self, env_pms, agent_pms, n_stp_max, pms):

        self.n_stp_max   = n_stp_max
        self.buff_size   = pms.buff_size
        self.n_buff      = pms.n_buff
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs
        self.mem_size    = self.n_buff * self.buff_size
        self.freq_report = max(int(n_stp_max / (freq_report * self.buff_size)), 1)

        # Initialize environment
        self.env = environment(paths.base, env_pms)

        # Initialize agent
        n_cpu = mpi.size*self.env.spaces.natural_act_dim()
        self.agent = agent_factory.create(agent_pms.type,
                                          spaces   = self.env.spaces,
                                          n_cpu    = n_cpu,
                                          mem_size = self.mem_size,
                                          pms      = agent_pms)

        # Initialize update
        self.update = update_factory.create("on_policy")

        # Initialize learning data report
        self.report = report(self.freq_report, ["step", "episode", "score", "smooth_score"])

        self.monitoring = False
        if hasattr(pms, "monitoring"): self.monitoring = pms.monitoring

        # Initialize counter
        self.counter = counter(mpi.size)

        # Initialize renderer
        self.rnd_style = "rgb_array"
        if hasattr(pms, "rnd_style"): self.rnd_style = pms.rnd_style
        self.renderer = renderer(mpi.size, self.rnd_style, pms.render_every)

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_training = timer("training ")

    def loop(self):

        # Local variables to avoid cumbersome expressions
        natural_act_dim = self.env.spaces.natural_act_dim()
        true_obs_dim    = self.env.spaces.true_obs_dim()

        self.timer_global.tic()
        obs = np.zeros((natural_act_dim, mpi.size, true_obs_dim))
        for i in range(natural_act_dim):
            obs[i,:,:] = self.env.reset_all()
        self.counter.reset()

        # Loop until max number of steps is reached
        while self.counter.step < self.n_stp_max:

            # Prepare inner training loop
            self.agent.pre_loop()

            # Loop over buff size
            while not (self.agent.buff.size() >= self.buff_size):

                act = np.zeros((natural_act_dim, mpi.size))
                nxt = np.zeros((natural_act_dim, mpi.size, true_obs_dim))
                rwd = np.zeros((natural_act_dim, mpi.size))
                lgp = np.zeros((natural_act_dim, mpi.size))
                dne = np.zeros((natural_act_dim, mpi.size), dtype=bool)
                trc = np.zeros((natural_act_dim, mpi.size), dtype=bool)

                # Get actions
                for i in range(natural_act_dim):
                    actions, log_prob = self.agent.actions(obs[i,:,:])
                    act[i,:] = np.reshape(actions, (mpi.size))
                    lgp[i,:] = log_prob

                # Perform a real step
                for i in range(natural_act_dim):
                    n, r, d, t = self.env.step(np.transpose(act))
                    nxt[i,:,:] = n[:,:]
                    rwd[i,:]   = r[:]
                    dne[i,:]   = d[:]
                    trc[i,:]   = t[:]

                # Terminate trajectories
                dne = np.reshape(dne, (natural_act_dim*mpi.size))
                trc = np.reshape(trc, (natural_act_dim*mpi.size))
                trm = self.agent.terminate(dne, trc)

                # Save reward for counter update
                counter_rwd = np.sum(rwd, axis=0)

                # Loop over local sub-envs
                act = np.reshape(act, (natural_act_dim*mpi.size))
                obs = np.reshape(obs, (natural_act_dim*mpi.size, true_obs_dim))
                nxt = np.reshape(nxt, (natural_act_dim*mpi.size, true_obs_dim))
                rwd = np.reshape(rwd, (natural_act_dim*mpi.size))
                lgp = np.reshape(lgp, (natural_act_dim*mpi.size))
                trm = np.reshape(trm, (natural_act_dim*mpi.size))

                self.agent.store(obs, nxt, act, lgp, rwd, trm)

                # Update counter
                self.counter.update(counter_rwd)

                # Handle rendering
                self.renderer.store(self.env)

                # Finish if some episodes are done
                dne = np.reshape(dne, (natural_act_dim, mpi.size))
                for cpu in range(mpi.size):
                    if dne[0,cpu]:
                        for _ in range(self.counter.ep_step[cpu]):
                            self.report.store(cpu=cpu, counter=self.counter)
                            self.counter.step += 1

                        self.print_episode()
                        self.renderer.finish(paths.run, self.counter.ep, cpu)

                        best = self.counter.reset_ep(cpu)
                        if best: self.agent.save_policy(paths.run + "/" + self.agent.name)

                # Update observation
                obs = nxt

                # Reset only finished environments
                obs = np.reshape(obs, (natural_act_dim, mpi.size, true_obs_dim))
                for i in range(natural_act_dim):
                    obs[i,:,:] = self.env.reset(dne[i,:], obs[i,:,:])

            # Finalize inner training loop
            self.agent.post_loop(style="buffer")

            # Write report data to file
            self.report.write(paths.run)

            # Train agent
            self.timer_training.tic()
            btc_size = math.floor(self.mem_size * self.btc_frac)
            self.update.update(self.agent, self.mem_size, btc_size, self.n_epochs)
            self.timer_training.toc()

        self.end_training()
