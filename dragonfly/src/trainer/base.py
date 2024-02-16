# Generic imports
import os
import shutil
from typing import Any

# Custom imports
from dragonfly.src.core.constants   import *
from dragonfly.src.utils.timer      import *
from dragonfly.src.env.environments import *
from dragonfly.src.agent.agent      import *
from dragonfly.src.update.update    import *
from dragonfly.src.utils.buff       import *
from dragonfly.src.utils.report     import *
from dragonfly.src.utils.renderer   import *
from dragonfly.src.utils.counter    import *
from dragonfly.src.utils.error      import *

class base_trainer():
    def __init__(
            self,
            env_pms: Any,
            path: str,
            n_stp_max: int,
            pms: Any
        ):
        """
        Initializes the base trainer class.

        Args:
            env_pms (Any): Parameters for the environment.
            path (str): Path for saving or loading data.
            n_stp_max (int): Maximum number of steps.
            pms (Any): Parameters for the trainer.
        """
        self.env = environments(path, env_pms, n_stp_max)
        self.obs_dim     = self.env.obs_dim
        self.act_dim     = self.env.act_dim
        self.n_stp_max   = n_stp_max

        self.agent = None

        if hasattr(pms, "update"): self.update_type = pms.update

        self.monitoring = False
        if hasattr(pms, "monitoring"): self.monitoring = pms.monitoring

        # Initialize counter
        self.counter = counter(mpi.size)

        # Initialize renderer
        self.rnd_style = "rgb_array"
        if hasattr(pms, "rnd_style"):
            self.rnd_style = pms.rnd_style
        self.renderer = renderer(mpi.size, self.rnd_style, pms.render_every)

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_training = timer("training ")

    def loop(self, path, run):
        raise NotImplementedError

    def monitor(self, path, run, obs, act):
        """
        Monitors and logs the actions and observations during training.
        This method handles the logging of actions and observations for each episode
        and step during the training process. 

        Args:
            path (str): The base path for logging.
            run (int): The current run number.
            obs (np.array): The observation array from the environment.
            act (np.array): The action array taken by the agent.

        Raises:
            error: If monitoring is attempted in a parallel environment setup.
        """
        if self.monitoring:
            if (mpi.size > 1):
                error("base_trainer", "monitor",
                      "monitoring does not work with parallel environments")

            apath = path+"/"+str(run)+"/actions"
            opath = path+"/"+str(run)+"/observations"

            if ((self.counter.ep         == 0) and
                (self.counter.ep_step[0] == 0)):
                if (os.path.exists(apath)): shutil.rmtree(apath)
                if (os.path.exists(opath)): shutil.rmtree(opath)
                os.makedirs(apath)
                os.makedirs(opath)

            # Write actions
            s = str(self.counter.ep_step[0])
            a = act.flatten()
            for i in range(act.size):
                s += " "
                s += str(a[i])
            s += "\n"

            filename = apath+"/"+str(self.counter.ep)+".dat"
            with open(filename, "a") as f:
                f.write(s)

            # Write observations
            s = str(self.counter.ep_step[0])
            o = obs.flatten()
            for i in range(obs.size):
                s += " "
                s += str(o[i])
            s += "\n"

            filename = opath+"/"+str(self.counter.ep)+".dat"
            with open(filename, "a") as f:
                f.write(s)

    def apply_next_step(self,
                        obs: np.array,
                        path: str,
                        run: int
        ):
        """
        Executes the next step in the environment using the current observation, updates the agent's state,
        and handles monitoring and rendering.

        This method sends the current observation to the agent to decide on an action, then applies this action
        in the environment to get the next state and reward. It stores the transition (current state, action,
        next state, reward, done flag, and trace) in the agent's memory. It also updates the training counter
        with the received reward and handles the rendering of the environment's current state.

        Args:
            obs (np.array): The current observation from the environment.
            path (str): The base path for logging and monitoring.
            run (int): The current run number.

        Returns:
            tuple: A tuple containing the next state (np.array), reward (float), done flag (bool),
        """
        act = self.agent.actions(obs)
        nxt, rwd, dne, trc = self.env.step(act)

        # Store transition
        self.agent.store(obs, nxt, act, rwd, dne, trc)
        self.monitor(path, run, obs, act)
        # Update counter
        self.counter.update(rwd)
        # Handle rendering
        self.renderer.store(self.env)
        return nxt, rwd, dne, trc

    def reset(self):
        self.agent.reset()
        self.report.reset()
        self.renderer.reset()

    def print_episode(self):
        """
        Prints a summary at the end of each episode.

        This method outputs the episode number, current step, average score, and the best score
        achieved so far, along with the episode number where the best score was achieved. It
        manages the printing format to ensure that the output is updated in place until the
        maximum number of steps is reached, after which it starts printing on new lines.
        """
        # No initial printing
        if (self.counter.ep == 0): return

        ep        = self.counter.ep
        stp       = self.counter.step
        n_stp_max = self.n_stp_max

        # Retrieve data
        avg    = self.report.avg("score", n_smooth)
        avg    = f"{avg:.3f}"
        bst    = self.counter.best_score
        bst    = f"{bst:.3f}"
        bst_ep = self.counter.best_ep

        # Handle no-printing after max step
        if (stp < n_stp_max-1):
            end = "\r"
            self.cnt = 0
        else:
            end  = "\n"
            self.cnt += 1

        if (self.cnt <= 1):
            print("# Ep #"+str(ep)+", step = "+str(stp)+", avg score = "+str(avg)+", best score = "+str(bst)+" at ep "+str(bst_ep)+"                 ", end=end)

    def store_report(self, cpu):
        for i in range(self.counter.ep_step[cpu]):
            if (self.counter.step%step_report == 0):
                self.report.append("step",    self.counter.step)
                self.report.append("episode", self.counter.ep)
                self.report.append("score",   self.counter.score[cpu])
                smooth_score = self.report.avg("score", n_smooth)
                self.report.append("smooth_score",  smooth_score)
            self.counter.step += 1

    def write_report(self, path, run, force=False):
        # Set filename with method name and run number
        filename = path+'/'+str(run)+'/'+str(run)+'.dat'
        self.report.write(filename, force)

    def end_training(self, path, run):
        """
        Finalizes a training by printing a summary, writing reports, and closing timers.

        This method is called at the end of training to perform final housekeeping tasks:
        it prints a summary of the episode, writes the episode's data to a report, and closes
        all active timers, displaying their summaries. This ensures that all relevant information
        is logged and that resources are properly managed at the conclusion of a training.

        Args:
            path (str): The base path for logging and monitoring.
            run (int): The current run number.
        """
        # Last printing
        self.print_episode()
        # Last writing
        self.write_report(path, run, force=True)
        # Close timers and show
        self.timer_global.toc()
        self.timer_global.show()
        self.env.timer_env.show()
        self.agent.timer_actions.show()
        self.timer_training.show()
