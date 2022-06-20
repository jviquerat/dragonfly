# Generic imports
import math
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

    # Reset
    def reset(self):

        self.agent.reset()
        self.report.reset()
        self.renderer.reset()

    # Printings at the end of an episode
    def print_episode(self):

        # No initial printing
        if (self.agent.counter.ep == 0): return

        # Average and print
        ep       = self.agent.counter.ep
        n_ep_max = self.n_ep_max

        if (ep <= n_ep_max):
            avg    = self.report.avg("score", n_smooth)
            avg    = f"{avg:.3f}"
            bst    = self.agent.counter.best_score
            bst    = f"{bst:.3f}"
            bst_ep = self.agent.counter.best_ep
            end    = "\n"
            if (ep < n_ep_max): end = "\r"
            stp    = self.agent.counter.step

            print("# Ep #"+str(ep)+", step = "+str(stp)+", avg score = "+str(avg)+", best score = "+str(bst)+" at ep "+str(bst_ep)+"                 ", end=end)

    ################################
    ### Report wrappings
    ################################

    # Store data in report
    def store_report(self, cpu):

        self.report.append("episode",       self.agent.counter.ep)
        self.report.append("step",          self.agent.counter.step)
        self.report.append("score",         self.agent.counter.score[cpu])
        smooth_score = self.report.avg("score", n_smooth)
        self.report.append("smooth_score",  smooth_score)
        self.report.append("length",        self.agent.counter.ep_step[cpu])
        smooth_length = self.report.avg("length", n_smooth)
        self.report.append("smooth_length", smooth_length)

    # Write learning data report
    def write_report(self, path, run):

        # Set filename with method name and run number
        filename = path+'/'+str(run)+'/'+str(run)+'.dat'
        self.report.write(filename)
