# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.core.constants        import *
from dragonfly.src.terminator.terminator import *
from dragonfly.src.utils.timer           import *
from dragonfly.src.utils.buff            import *
from dragonfly.src.utils.report          import *
from dragonfly.src.utils.renderer        import *
from dragonfly.src.utils.counter         import *

###############################################
### Class for buffer-based training
### obs_dim     : dimension of observations
### act_dim     : dimension of actions
### pol_act_dim : true dimension of the actions provided to the env
### n_cpu       : nb of parallel environments
### n_ep_max    : max nb of episodes to unroll in a run
### pms         : parameters
class buffer_based():
    def __init__(self, obs_dim, act_dim,
                 pol_act_dim, n_cpu, n_ep_max, pms):

        # Initialize from input
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.pol_act_dim = pol_act_dim
        self.n_cpu       = n_cpu
        self.n_ep_max    = n_ep_max
        self.buff_size   = pms.buff_size
        self.n_buff      = pms.n_buff
        self.btc_frac    = pms.batch_frac
        self.n_epochs    = pms.n_epochs

        # pol_act_dim is the true dimension of the action provided to the env
        # This allows compatibility between continuous and discrete envs
        self.loc_buff = loc_buff(self.n_cpu,
                                 self.obs_dim,
                                 self.pol_act_dim)
        self.glb_buff = glb_buff(self.n_cpu,
                                 self.obs_dim,
                                 self.pol_act_dim,
                                 self.n_buff,
                                 self.buff_size,
                                 self.btc_frac)

        # Initialize learning data report
        self.report_fields = ["episode",
                              "score",  "smooth_score",
                              "length", "smooth_length",
                              "step"]
        self.report = report(self.report_fields)

        # Initialize renderer
        self.renderer = renderer(self.n_cpu, pms.render_every)

        # Initialize counter
        self.counter = counter(self.n_cpu,
                               self.n_ep_max,
                               "buffer",
                               buff_size=self.buff_size)

        # Initialize terminator
        self.terminator = terminator_factory.create(pms.terminator.type,
                                                    n_cpu = self.n_cpu,
                                                    pms   = pms.terminator)

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_env      = timer("env      ")
        self.timer_actions  = timer("actions  ")
        self.timer_training = timer("training ")

    # Loop
    def loop(self, path, run, env, agent):

        # Start global timer
        self.timer_global.tic()

        # Reset environment
        obs = env.reset_all()

        # Loop until max episode number is reached
        while (not self.counter.done_max_ep()):

            # Reset local buffer
            self.loc_buff.reset()

            # Loop over buff size
            while (not self.counter.done_buffer(self.loc_buff)):

                # Get actions
                self.timer_actions.tic()
                act = agent.get_actions(obs)
                self.timer_actions.toc()

                # Make one env step
                self.timer_env.tic()
                nxt, rwd, done = env.step(act)
                self.timer_env.toc()

                # Handle termination state
                trm, bts, epn = self.terminator.terminate(self.counter, done)

                # Store transition
                self.loc_buff.store(obs, nxt, act, rwd, trm, bts, epn)

                # Update counter
                self.counter.update(rwd)

                # Handle rendering
                rnd = env.render(self.renderer.render)
                self.renderer.store(rnd)

                # Finish if some episodes are done
                self.finish_episodes(path, done)

                # Update observation
                obs = nxt

                # Reset only finished environments
                self.timer_env.tic()
                env.reset(done, obs)
                self.timer_env.toc()

            # Finalize buffers for training
            self.terminator.bootstrap_terminal(self.loc_buff)
            obs, nxt, act, rwd, trm, bts, epn = self.loc_buff.serialize()
            tgt, adv = agent.compute_returns(obs, nxt, act, rwd, trm, bts)

            # Store in global buffers
            self.glb_buff.store(obs, adv, tgt, act, epn)

            # Train agent
            self.timer_training.tic()
            self.train(agent)
            self.timer_training.toc()

            # Write report data to file
            self.write_report(agent, self.report, path, run)

        # Last printing
        self.print_episode(self.counter, self.report)

        # Close timers and show
        self.timer_global.toc()
        self.timer_global.show()
        self.timer_env.show()
        self.timer_actions.show()
        self.timer_training.show()

    # Train
    def train(self, agent):

        # Save previous policy
        agent.policy.save_prv()

        # Train policy and v_value
        for epoch in range(self.n_epochs):

            # Retrieve data
            obs, act, adv, tgt = self.glb_buff.get_buff()
            done               = False

            # Visit all available history
            while not done:
                start, end, done = self.glb_buff.get_indices()
                btc_obs          = obs[start:end]
                btc_act          = act[start:end]
                btc_adv          = adv[start:end]
                btc_tgt          = tgt[start:end]

                agent.train(btc_obs, btc_act, btc_adv, btc_tgt, end-start)

    # Reset
    def reset(self):

        self.loc_buff.reset()
        self.glb_buff.reset()
        self.report.reset(self.report_fields)
        self.renderer.reset()
        self.counter.reset()

    # Finish if some episodes are done
    def finish_episodes(self, path, done):

        # Loop over environments and finalize/reset
        for cpu in range(self.n_cpu):
            if (done[cpu]):
                self.store_report(self.counter, self.report, cpu)
                self.print_episode(self.counter, self.report)
                self.renderer.finish(path, self.counter.ep, cpu)
                self.counter.reset_ep(cpu)

    # Printings at the end of an episode
    def print_episode(self, counter, report):

        # No initial printing
        if (counter.get_ep() == 0): return

        # Average and print
        if (counter.get_ep() <= counter.get_n_ep_max()):
            avg    = report.avg_score(n_smooth)
            avg    = f"{avg:.3f}"
            bst    = counter.get_best_score()
            bst    = f"{bst:.3f}"
            bst_ep = counter.get_best_ep()
            end    = '\n'
            if (counter.get_ep() < counter.get_n_ep_max()): end = '\r'
            print('# Ep #'+str(counter.get_ep())+', avg score = '+str(avg)+', best score = '+str(bst)+' at ep '+str(bst_ep)+'                 ', end=end)

    ################################
    ### Report wrappings
    ################################

    # Store data in report
    def store_report(self, counter, report, cpu):

        report.append("episode",       counter.ep)
        report.append("score",         counter.score[cpu])
        smooth_score   = np.mean(report.data["score"][-n_smooth:])
        report.append("smooth_score",  smooth_score)
        report.append("length",        counter.ep_step[cpu])
        smooth_length  = np.mean(report.data["length"][-n_smooth:])
        report.append("smooth_length", smooth_length)

        report.step(counter.ep_step[cpu])

    # Write learning data report
    def write_report(self, agent, report, path, run):

        # Set filename with method name and run number
        filename = path+'/'+agent.name+'_'+str(run)+'.dat'
        report.write(filename, self.report_fields)
