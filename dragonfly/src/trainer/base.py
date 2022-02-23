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
    def loop(self, path, run, env, agent):
        raise NotImplementedError

    # Finish if some episodes are done
    def finish_episodes(self, path, done):
        raise NotImplementedError

    # Train
    def train(self, agent):
        raise NotImplementedError

    # Reset
    def reset(self):

        self.buff.reset()
        self.gbuff.reset()
        self.report.reset()
        self.renderer.reset()
        self.counter.reset()

    # Printings at the end of an episode
    def print_episode(self, counter, report):

        # No initial printing
        if (counter.get_ep() == 0): return

        # Average and print
        if (counter.get_ep() <= counter.get_n_ep_max()):
            avg    = report.avg("score", n_smooth)
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

        n_step = counter.ep_step[cpu]
        if (counter.ep > 0): n_step += report.get("step")[-1]

        report.append("episode",       counter.ep)
        report.append("step",          n_step)
        report.append("score",         counter.score[cpu])
        smooth_score = report.avg("score", n_smooth)
        report.append("smooth_score",  smooth_score)
        report.append("length",        counter.ep_step[cpu])
        smooth_length = report.avg("length", n_smooth)
        report.append("smooth_length", smooth_length)

    # Write learning data report
    def write_report(self, agent, report, path, run):

        # Set filename with method name and run number
        filename = path+'/'+str(run)+'/'+agent.name+'_'+str(run)+'.dat'
        report.write(filename)
