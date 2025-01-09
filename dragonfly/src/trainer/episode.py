# Custom imports
from dragonfly.src.trainer.base import *

###############################################
### Episode-based trainer class
class episode(base_trainer):
    def __init__(self, env_pms, agent_pms, n_stp_max, pms):

        self.n_stp_max   = n_stp_max
        self.n_ep_unroll = pms.n_ep_unroll*mpi.size
        self.n_ep_train  = pms.n_ep_train
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs
        self.mem_size    = 1000*self.n_ep_train
        self.freq_report = 10

        self.lengths   = np.array([], dtype=int)
        self.ep_unroll = 0

        super().__init__("on_policy", env_pms, agent_pms, pms)

    def loop(self):

        self.timer_global.tic()
        obs = self.env.reset_all()
        self.counter.reset()

        # Loop until max number of steps is reached
        while self.counter.step < self.n_stp_max:

            # Prepare inner training loop
            self.agent.pre_loop()

            # Loop over training episodes
            while not (self.ep_unroll >= self.n_ep_unroll):

                # Retrieve action and step
                act, lgp = self.agent.actions(obs)
                nxt, rwd, dne, trc = self.env.step(act)

                # Terminate trajectories
                trm = self.agent.terminate(dne, trc)

                # Store transition
                self.agent.store(obs, nxt, act, lgp, rwd, trm)
                self.monitor(obs, act)

                # Update counter
                self.counter.update(rwd)

                # Handle rendering
                self.renderer.store(self.env)

                # Finish if some episodes are done
                for cpu in range(mpi.size):
                    if dne[cpu]:
                        self.lengths = np.append(self.lengths, self.counter.ep_step[cpu])

                        for _ in range(self.counter.ep_step[cpu]):
                            self.report.store(cpu=cpu, counter=self.counter)
                            self.counter.step += 1

                        self.print_episode()
                        self.renderer.finish(paths.run, self.counter.ep, cpu)

                        best = self.counter.reset_ep(cpu)
                        if best: self.agent.save_policy(paths.run + "/" + self.agent.name)

                        self.ep_unroll += 1

                # Update observation
                obs = nxt

                # Reset only finished environments
                self.env.reset(dne, obs)

            # Finalize inner training loop
            self.agent.post_loop()

            # Write report data to file
            self.report.write(paths.run)

            # Train agent
            self.timer_training.tic()
            size = np.sum(self.lengths[-self.n_ep_train :])
            btc_size = math.floor(size * self.btc_frac)
            self.update.update(self.agent, size, btc_size, self.n_epochs)
            self.timer_training.toc()

            # Reset unroll
            self.ep_unroll = 0

        self.end_training()
