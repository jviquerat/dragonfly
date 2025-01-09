# Custom imports
from dragonfly.src.trainer.base import *

###############################################
### Buffer-based trainer class
class buffer(base_trainer):
    def __init__(self, env_pms, agent_pms, n_stp_max, pms):

        self.n_stp_max   = n_stp_max
        self.buff_size   = pms.buff_size
        self.n_buff      = pms.n_buff
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs
        self.mem_size    = self.n_buff * self.buff_size
        self.freq_report = max(int(n_stp_max / (freq_report * self.buff_size)), 1)

        super().__init__("on_policy", env_pms, agent_pms, pms)

    def loop(self):

        self.timer_global.tic()
        obs = self.env.reset_all()
        self.counter.reset()

        # Loop until max number of steps is reached
        while self.counter.step < self.n_stp_max:

            # Prepare inner training loop
            self.agent.pre_loop()

            # Loop over buff size
            while not (self.agent.buff.size() >= self.buff_size):

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
                obs = self.env.reset(dne, obs)

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
