# Generic imports
import numpy as np

###############################################
### Counter, a small util to count episode steps and scores
### n_cpu : nb of parallel environements
### n_ep  : max nb of episodes in a run
class counter:
    def __init__(self, n_cpu, n_ep):

        self.n_cpu   = n_cpu
        self.n_ep    = n_ep
        self.reset()

    # Reset
    def reset(self):

        self.ep         =  0
        self.best_ep    =  0
        self.best_score =-1.0e8
        self.ep_step    = [0     for _ in range(self.n_cpu)]
        self.score      = [0.0   for _ in range(self.n_cpu)]

    # Test total nb of episodes
    def max_total_ep(self):

        return (self.ep >= self.n_ep)

    # Update score
    def update_score(self, rwd):

        self.score[:] += rwd[:]

    # Update step
    def update_step(self):

        self.ep_step[:] = [x+1 for x in self.ep_step]

    # Reset episode counters and update best values
    def reset_ep(self, cpu):

        if (self.score[cpu] >= self.best_score):
            self.best_score = self.score[cpu]
            self.best_ep    = self.ep

        self.score[cpu]   = 0
        self.ep_step[cpu] = 0
        self.ep          += 1


