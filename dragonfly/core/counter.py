# Generic imports
import numpy as np

###############################################
### Counter, a small util to count episode steps and scores
class counter:
    def __init__(self, n_cpu, n_ep):

        # Initialize
        self.n_cpu   = n_cpu
        self.n_ep    = n_ep
        self.ep      =  0
        self.ep_step = [0     for _ in range(n_cpu)]
        self.score   = [0.0   for _ in range(n_cpu)]

    # Test episode loop
    def test_ep_loop(self):

        return (self.ep < self.n_ep)

    # Update score
    def update_score(self, rwd):

        self.score[:] += rwd[:]

    # Update step
    def update_step(self):

        self.ep_step[:] = [x+1 for x in self.ep_step]

    # Reset episode counter
    def reset_ep(self, cpu):

        self.score[cpu]   = 0
        self.ep_step[cpu] = 0
        self.ep          += 1

