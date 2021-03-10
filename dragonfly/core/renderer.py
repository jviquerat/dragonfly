# Generic imports
import numpy as np

###############################################
### Renderer, used to store rendering returns from gym envs
class renderer:
    def __init__(self, n_cpu):

        # Initialize
        self.n_cpu        = n_cpu
        self.rgb          = [[] for _ in range(n_cpu)]
