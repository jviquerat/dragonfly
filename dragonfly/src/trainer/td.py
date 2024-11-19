# Custom imports
from dragonfly.src.trainer.base import *

###############################################
### TD-based trainer class
class td(base_trainer):
    def __init__(self, env_pms, agent_pms, n_stp_max, pms):

        self.n_stp_max         = n_stp_max
        self.mem_size          = pms.mem_size
        self.n_stp_unroll      = pms.n_stp_unroll
        self.n_true_stp_unroll = pms.n_stp_unroll * mpi.size
        self.btc_size          = pms.btc_size
        self.freq_report       = max(int(self.n_stp_max / (freq_report * self.n_true_stp_unroll)), 1)

        self.stp_unroll = 0

        super().__init__("off_policy", env_pms, agent_pms, pms)

    def loop(self):

        self.timer_global.tic()
        obs = self.env.reset_all()
        self.counter.reset()

        # Loop until max number of steps reached
        while self.counter.step < self.n_stp_max:

            # Prepare inner training loop
            self.agent.pre_loop()

            # Loop over training steps
            while not (self.stp_unroll >= self.n_true_stp_unroll):
                nxt, _, dne, _ = self.apply_next_step(obs)

                # Update unrolling counter
                self.stp_unroll += mpi.size

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
                self.env.reset(dne, obs)

            # Finalize inner training loop
            self.agent.post_loop()

            # Write report data to file
            self.report.write(paths.run)

            # Train agent
            self.timer_training.tic()
            self.update.update(self.agent, self.btc_size, self.n_stp_unroll)
            self.timer_training.toc()

            # Reset unroll
            self.stp_unroll = 0

        self.end_training()
