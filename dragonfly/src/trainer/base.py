# Generic imports
import os
import math
import shutil
import numpy as np

# Custom imports
from dragonfly.src.core.constants    import *
from dragonfly.src.utils.timer       import *
from dragonfly.src.envs.environments import *
from dragonfly.src.agent.agent       import *
from dragonfly.src.update.update     import *
from dragonfly.src.utils.buff        import *
from dragonfly.src.utils.report      import *
from dragonfly.src.utils.renderer    import *
from dragonfly.src.utils.counter     import *
from dragonfly.src.utils.error       import *

###############################################
### Class for buffer-based training
### obs_dim  : dimension of observations
### act_dim  : dimension of actions
### pol_dim  : true dimension of the actions provided to the env
### n_cpu    : nb of parallel environments
### n_ep_max : max nb of episodes to unroll in a run
### pms      : parameters
class trainer_base():
    def __init__(self):
        pass

    # Loop
    def loop(self, path, run):
        raise NotImplementedError

    # Finish if some episodes are done
    def finish_episodes(self, path, done):
        raise NotImplementedError

    # Train
    def train(self):
        raise NotImplementedError

    # Monitor transition
    def monitor(self, path, run, obs, act):
        if self.monitoring:
            # Handle folder
            fpath = path+"/"+str(run)+"/actions"
            if ((self.counter.step == 0)):
                if (os.path.exists(fpath)): shutil.rmtree(fpath)
                os.makedirs(fpath)

            # Write actions
            s = str(self.counter.ep_step[0])
            for i in range(act.size):
                s += " "
                s += str(act.flatten()[i])
            s += "\n"
            filename = fpath+"/"+str(self.counter.ep)+"_act.dat"
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
