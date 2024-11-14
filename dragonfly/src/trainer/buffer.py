# Custom imports
from dragonfly.src.trainer.base import *

###############################################
### Buffer-based trainer class
class buffer(base_trainer):
    def __init__(self, env_pms, agent_pms, n_stp_max, pms):
        super().__init__(env_pms, n_stp_max, pms)

        self.buff_size = pms.buff_size
        self.n_buff = pms.n_buff
        self.btc_frac = pms.batch_frac
        self.n_epochs = pms.n_epochs
        self.size = self.n_buff * self.buff_size
        self.freq_report = max(int(n_stp_max / (freq_report * self.buff_size)), 1)
        self.update_type = "on_policy"

        self.warmup = 0
        if hasattr(pms, "warmup"): self.warmup = pms.warmup

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.type,
                                          spaces = self.env.spaces,
                                          n_cpu  = mpi.size,
                                          size   = self.size,
                                          pms    = agent_pms)

        # Initialize update
        self.update = update_factory.create(self.update_type)

        # Initialize learning data report
        self.report = report(
            self.freq_report, ["step", "episode", "score", "smooth_score"]
        )

    def loop(self):

        # Loop until max episode number is reached
        obs = self.start_training()
        while self.counter.step < self.n_stp_max:

            # Prepare inner training loop
            self.agent.pre_loop()

            # Loop over buff size
            while not (self.agent.buff.size() >= self.buff_size):
                nxt, _, dne, _ = self.apply_next_step(obs)

                # Finish if some episodes are done
                for cpu in range(mpi.size):
                    if dne[cpu]:
                        for _ in range(self.counter.ep_step[cpu]):
                            self.report.store(cpu=cpu, counter=self.counter)
                            self.counter.step += 1

                        self.print_episode()
                        self.renderer.finish(paths.run, self.counter.ep, cpu)
                        best = self.counter.reset_ep(cpu)
                        name = paths.run + "/" + self.agent.name
                        if best:
                            self.agent.save_policy(name)

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
            btc_size = math.floor(self.size * self.btc_frac)

            if (self.counter.step > self.warmup):
                self.update.update(self.agent,
                                   self.size, btc_size, self.n_epochs)

            self.timer_training.toc()

        self.end_training()
