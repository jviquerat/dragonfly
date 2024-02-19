from dragonfly.src.trainer.base import *


class buffer(base_trainer):
    def __init__(self, env_pms, agent_pms, path, n_stp_max, pms):
        """
        Args:
            env_pms (Any): Parameters for the environment.
            agent_pms (Any): Parameters for the agent.
            path (str): Path for saving or loading data.
            n_stp_max (int): Maximum number of steps.
            pms (Any): Parameters for the trainer.
        """
        super().__init__(env_pms=env_pms, path=path, n_stp_max=n_stp_max, pms=pms)

        self.buff_size = pms.buff_size
        self.n_buff = pms.n_buff
        self.btc_frac = pms.batch_frac
        self.n_epochs = pms.n_epochs
        self.size = self.n_buff * self.buff_size
        self.freq_report = max(int(n_stp_max / (freq_report * self.buff_size)), 1)
        self.update_type = "on_policy"

        # Initialize agent
        self.agent = agent_factory.create(
            agent_pms.type,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            n_cpu=mpi.size,
            size=self.size,
            pms=agent_pms,
        )

        # Initialize update
        self.update = update_factory.create(self.update_type)

        # Initialize learning data report
        self.report = report(
            self.freq_report, ["step", "episode", "score", "smooth_score"]
        )

    def loop(self, path, run):
        """
        Executes the main training loop until the maximum number of steps is reached.

        This method orchestrates the training process by continuously executing steps in the environment,
        updating observations, and managing the buffer. It handles the training loop, including preparing
        the agent for training, executing steps until the buffer is filled, processing completed episodes,
        and training the agent with the collected data. The loop continues until the predefined maximum
        number of steps is reached. It concludes by finalizing the training session.

        Args:
            path (str): The base path for saving models and reports.
            run (int): The current run identifier.
        """
        obs = self.start_training()
        # Loop until max episode number is reached
        while self.counter.step < self.n_stp_max:
            # Prepare inner training loop
            self.agent.pre_loop()
            # Loop over buff size
            while not (self.agent.buff.size() >= self.buff_size):
                nxt, _, dne, _ = self.apply_next_step(obs, path, run)
                # Finish if some episodes are done
                for cpu in range(mpi.size):
                    if dne[cpu]:
                        self.store_report(cpu)
                        self.print_episode()
                        self.renderer.finish(path, run, self.counter.ep, cpu)
                        best = self.counter.reset_ep(cpu)
                        name = path + "/" + str(run) + "/" + self.agent.name
                        if best:
                            self.agent.save(name)
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
            btc_size = math.floor(self.size * self.btc_frac)
            self.update.update(self.agent, self.size, btc_size, self.n_epochs)
            self.timer_training.toc()

        self.end_training(path, run)
