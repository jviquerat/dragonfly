from dragonfly.src.trainer.base import *


class td(base_trainer):
    def __init__(self, env_pms, agent_pms, path, n_stp_max, pms):
        """
        Args:
            env_pms (Any): Parameters for the environment.
            agent_pms (Any): Parameters for the agent.
            path (str): Path for saving or loading data.
            n_stp_max (int): Maximum number of steps.
            pms (Any): Parameters for the trainer.
        """
        super().__init__(self, env_pms=env_pms, path=path, n_stp_max=n_stp_max)

        self.mem_size = pms.mem_size
        self.n_stp_unroll = pms.n_stp_unroll * mpi.size
        self.btc_size = pms.btc_size
        self.freq_report = max(
            int(self.n_stp_max / (freq_report * self.n_stp_unroll)), 1
        )
        self.update_type = "off_policy"

        # Local variables
        self.unroll = 0

        # Initialize agent
        self.agent = agent_factory.create(
            agent_pms.type,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            n_cpu=mpi.size,
            size=self.mem_size,
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
        Executes the training loop for temporal difference learning until the maximum number of steps is reached.

        This method manages the training process for temporal difference (TD) learning. It starts with the initial
        observation and continues through the training steps until the maximum number of steps is reached. The loop
        involves preparing the agent for training, executing steps to progress through the training, handling the
        completion of episodes, and resetting environments as necessary. The training steps are unrolled in batches,
        and the agent is trained based on the collected data after each batch. The loop includes storing reports,
        printing episode summaries, saving the agent's state under certain conditions, and updating the observation
        for the next step. The process concludes with final training adjustments and reporting.

        Args:
            path (str): The base path for saving models, reports, and other data.
            run (int): The current run identifier.
        """
        obs = self.start_training()
        # Loop until max episode number is reached
        while self.counter.step < self.n_stp_max:
            # Prepare inner training loop
            self.agent.pre_loop()
            # Loop over training steps
            while not (self.unroll >= self.n_stp_unroll):
                nxt, _, dne, _ = self.apply_next_step(obs, path, run)
                # Update unrolling counter
                self.unroll += mpi.size
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
            self.agent.post_loop()
            # Write report data to file
            self.write_report(path, run)
            # Train agent
            self.timer_training.tic()
            self.update.update(self.agent, self.btc_size, self.n_stp_unroll)
            self.timer_training.toc()
            # Reset unroll
            self.unroll = 0
        self.end_training(path, run)
