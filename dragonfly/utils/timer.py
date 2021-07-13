# Generic imports
import time

###############################################
### timer class
### Used to measure time spent on different operations
class timer():
    def __init__(self, name):
        self.reset(name)

    def reset(self, name):
        self.name     = name
        self.time_tic = None
        self.time_toc = None
        self.dt       = 0.0

    def tic(self):
        self.t_tic = time.perf_counter()

    def toc(self, show=False):
        self.t_toc = time.perf_counter()
        self.dt   += self.t_toc - self.t_tic

        if (show): self.show()

    def show(self):
        str_dt = str(f"{self.dt:.2f}")
        print("# Timer "+str(self.name)+": "+str_dt+" seconds")
