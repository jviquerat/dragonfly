# Generic imports
import numpy as np

###############################################
### Counter, a small util to count episode steps and scores
### n_cpu       : nb of parallel environements
### n_ep_max    : max nb of episodes to unroll in a run
class counter:
    def __init__(self, n_cpu, n_ep_max):

        self.n_cpu    = n_cpu
        self.n_ep_max = n_ep_max

        self.reset()

    # Reset
    def reset(self):

        self.ep         =  0
        self.best_ep    =  0
        self.best_score = -1.0e8
        self.ep_step    = [0     for _ in range(self.n_cpu)]
        self.score      = [0.0   for _ in range(self.n_cpu)]
        self.entropy    = [0.0   for _ in range(self.n_cpu)]

    # Update score and step counter
    def update(self, rwd, entropy=[0.0]):

        self.score[:]  += rwd[:]
        self.ep_step[:] = [x+1 for x in self.ep_step]
        self.entropy[:] = entropy

    # Reset episode counters and update best values
    def reset_ep(self, cpu):

        best = False
        if (self.score[cpu] >= self.best_score):
            best            = True
            self.best_score = self.score[cpu]
            self.best_ep    = self.ep

        self.score[cpu]   = 0
        self.ep_step[cpu] = 0
        self.ep          += 1

        return best
