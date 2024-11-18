# Generic imports
import os
import shutil
import math
import numpy as np

# Custom imports
from dragonfly.src.core.constants   import *
from dragonfly.src.env.mpi          import mpi
from dragonfly.src.core.paths       import paths
from dragonfly.src.utils.timer      import timer
from dragonfly.src.env.environment  import environment
from dragonfly.src.agent.agent      import agent_factory
from dragonfly.src.update.update    import update_factory
from dragonfly.src.utils.report     import report
from dragonfly.src.utils.renderer   import renderer
from dragonfly.src.utils.counter    import counter
from dragonfly.src.utils.error      import error

###############################################
### Base trainer class
class base_trainer:
    def __init__(self, update_type, env_pms, agent_pms, pms):

        # Initialize environment
        self.env = environment(paths.base, env_pms)

        # Initialize agent
        self.agent = agent_factory.create(agent_pms.type,
                                          spaces   = self.env.spaces,
                                          n_cpu    = mpi.size,
                                          mem_size = self.mem_size,
                                          pms      = agent_pms)

        # Initialize update
        self.update = update_factory.create(update_type)

        # Initialize learning data report
        self.report = report(self.freq_report, ["step", "episode", "score", "smooth_score"])

        self.monitoring = False
        if hasattr(pms, "monitoring"): self.monitoring = pms.monitoring

        # Initialize counter
        self.counter = counter(mpi.size)

        # Initialize renderer
        self.rnd_style = "rgb_array"
        if hasattr(pms, "rnd_style"): self.rnd_style = pms.rnd_style
        self.renderer = renderer(mpi.size, self.rnd_style, pms.render_every)

        # Initialize timers
        self.timer_global   = timer("global   ")
        self.timer_training = timer("training ")

    # Mandatory re-implementation in daughter classes
    def loop(self):
        raise NotImplementedError

    # Execute next step in the environment
    def apply_next_step(self, obs):

        # Retrieve action and step
        act = self.agent.actions(obs)
        nxt, rwd, dne, trc = self.env.step(act)

        # Store transition
        self.agent.store(obs, nxt, act, rwd, dne, trc)
        self.monitor(obs, act)

        # Update counter
        self.counter.update(rwd)

        # Handle rendering
        self.renderer.store(self.env)
        return nxt, rwd, dne, trc

    # Reset trainer
    def reset(self):

        self.agent.reset()
        self.report.reset()
        self.renderer.reset()
        self.stop_print = False

    # Print summary at the end of each episode
    def print_episode(self):

        ep        = self.counter.ep
        stp       = self.counter.step
        n_stp_max = self.n_stp_max

        # No initial printing
        if ep == 0: return

        # No printing beyond final print
        if self.stop_print: return

        # Retrieve data
        avg = self.report.avg("score", n_smooth)
        avg = f"{avg:.3f}"
        bst = self.counter.best_score
        bst = f"{bst:.3f}"
        bst_ep = self.counter.best_ep

        # Handle no-printing after max step
        end = "\r"
        if (stp >= n_stp_max-1):
            end = "\n"
            self.stop_print = True

        print("# Ep #"
              + str(ep)
              + ", step = "
              + str(stp)
              + ", avg score = "
              + str(avg)
              + ", best score = "
              + str(bst)
              + " at ep "
              + str(bst_ep)
              + "                 ",
              end=end)

    # Finalizes a training by printing a summary, writing reports, and closing timers.
    def end_training(self):

        # Last printing
        self.print_episode()

        # Last writing
        self.report.write(paths.run, force=True)

        # Close timers and show
        self.timer_global.toc()
        self.timer_global.show()
        self.env.timer_env.show()
        self.agent.timer_actions.show()
        self.timer_training.show()

    # Monitor observations and actions during training
    def monitor(self, obs, act):

        if (not self.monitoring): return

        # Monitoring only works in serial mode
        if mpi.size > 1:
            error("base_trainer", "monitor",
                  "monitoring does not work with parallel environments")

        apath = paths.run + "/actions"
        opath = paths.run + "/observations"

        if (self.counter.ep == 0) and (self.counter.ep_step[0] == 0):
            if os.path.exists(apath):
                shutil.rmtree(apath)
            if os.path.exists(opath):
                shutil.rmtree(opath)
            os.makedirs(apath)
            os.makedirs(opath)

        # Write actions
        s = str(self.counter.ep_step[0])
        a = act.flatten()
        for i in range(act.size):
            s += " "
            s += str(a[i])
        s += "\n"

        filename = apath + "/" + str(self.counter.ep) + ".dat"
        with open(filename, "a") as f:
            f.write(s)

        # Write observations
        s = str(self.counter.ep_step[0])
        o = obs.flatten()
        for i in range(obs.size):
            s += " "
            s += str(o[i])
        s += "\n"

        filename = opath + "/" + str(self.counter.ep) + ".dat"
        with open(filename, "a") as f:
            f.write(s)
