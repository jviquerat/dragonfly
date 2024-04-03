from dragonfly.src.trainer.base import *


class episode(base_trainer):
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

        self.n_ep_unroll = pms.n_ep_unroll * (mpi.size)
        self.n_ep_train = pms.n_ep_train
        self.btc_frac = pms.batch_frac
        self.n_epochs = pms.n_epochs
        self.size = 1000 * self.n_ep_train
        self.freq_report = 10
        self.update_type = "on_policy"

        # Local variables
        self.lengths = np.array([], dtype=int)
        self.unroll = 0

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
        Executes the training loop for a specified number of episodes, managing the training process.

        This method oversees the training loop, starting with the initial observation and continuing
        until the maximum number of steps is reached. It involves preparing the agent for training,
        executing steps to progress through episodes, handling the completion of episodes, and
        resetting environments as necessary. The loop includes collecting data on episode lengths,
        storing reports, printing episode summaries, and saving the agent's state under certain
        conditions. Training episodes are processed in batches, and the agent is trained based on
        the collected data. The loop concludes with final training adjustments and reporting.

        Args:
            path (str): The base path for saving models, reports, and other data.
            run (int): The current run identifier.
        """
        obs = self.start_training()
        # Loop until max episode number is reached
        while self.counter.step < self.n_stp_max:
            # Prepare inner training loop
            self.agent.pre_loop()
            # Loop over training episodes
            while not (self.unroll >= self.n_ep_unroll):
                nxt, _, dne, _ = self.apply_next_step(obs, path, run)
                # Finish if some episodes are done
                for cpu in range(mpi.size):
                    if dne[cpu]:
                        self.lengths = np.append(
                            self.lengths, self.counter.ep_step[cpu]
                        )
                        for _ in range(self.counter.ep_step[cpu]):
                            self.report.store(cpu=cpu, counter=self.counter)
                            self.counter.step += 1
                        self.print_episode()
                        self.renderer.finish(path, run, self.counter.ep, cpu)
                        best = self.counter.reset_ep(cpu)
                        name = path + "/" + str(run) + "/" + self.agent.name
                        if best:
                            self.agent.save_policy(name)
                        self.unroll += 1
                # Update observation
                obs = nxt
                # Reset only finished environments
                self.env.reset(dne, obs)
            # Finalize inner training loop
            self.agent.post_loop()
            # Write report data to file
            self.report.write(path, run)
            # Train agent
            self.timer_training.tic()
            size = np.sum(self.lengths[-self.n_ep_train :])
            btc_size = math.floor(size * self.btc_frac)
            self.update.update(self.agent, size, btc_size, self.n_epochs)
            self.timer_training.toc()
            # Reset unroll
            self.unroll = 0

        self.end_training(path, run)
