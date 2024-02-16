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

###############################################
### Base trainer class
class base_trainer():
    def __init__(
            self,
            env_pms: Any,
            path: str,
            n_stp_max: int
        ):
        """
        Initializes the base trainer class.

        Args:
            env_pms (Any): Parameters for the environment.
            path (str): Path for saving or loading data.
            n_stp_max (int): Maximum number of steps.
        """
        self.env = environments(path, env_pms, n_stp_max)
        self.obs_dim     = self.env.obs_dim
        self.act_dim     = self.env.act_dim
        self.n_stp_max   = n_stp_max

    # Loop
    def loop(self, path, run):
        raise NotImplementedError

    # Monitor transition
    def monitor(self, path, run, obs, act):
        if self.monitoring:

            # Check nb of cpus
            if (mpi.size > 1):
                error("base_trainer", "monitor",
                      "monitoring does not work with parallel environments")

            # Handle folders
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

    # Reset
    def reset(self):

        self.agent.reset()
        self.report.reset()
        self.renderer.reset()

    # Printings at the end of an episode
    def print_episode(self):

        # No initial printing
        if (self.counter.ep == 0): return

        # Average and print
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

    ################################
    ### Report wrappings
    ################################

    # Store data in report
    def store_report(self, cpu):

        for i in range(self.counter.ep_step[cpu]):
            if (self.counter.step%step_report == 0):
                self.report.append("step",    self.counter.step)

                self.report.append("episode", self.counter.ep)
                self.report.append("score",   self.counter.score[cpu])
                smooth_score = self.report.avg("score", n_smooth)
                self.report.append("smooth_score",  smooth_score)
            self.counter.step += 1

    # Write learning data report
    def write_report(self, path, run, force=False):

        # Set filename with method name and run number
        filename = path+'/'+str(run)+'/'+str(run)+'.dat'
        self.report.write(filename, force)
