# Generic imports
import PIL
import numpy as np

###############################################
### Renderer, used to store rendering returns from gym envs
### n_cpu        : nb of parallel environments
### renver_every : rendering frequency (in timesteps)
class renderer:
    def __init__(self, n_cpu, render_every):

        self.n_cpu        = n_cpu
        self.render_every = render_every
        self.reset()

    # Reset
    def reset(self):

        self.rgb    = [[]    for _ in range(self.n_cpu)]
        self.render = [False for _ in range(self.n_cpu)]

    # Store one rendering step for all cpus
    def store(self, rnd):

        # Not all environment render simultaneously
        # We use a list to select those that render and those that don't
        for cpu in range(self.n_cpu):
            if (self.render[cpu]):
                img = PIL.Image.fromarray(rnd[cpu])
                self.rgb[cpu].append(img)

    # Finish rendering process, saving to gif
    def finish(self, path, ep, cpu):

        # Render if necessary
        if (self.render[cpu]):
            self.render[cpu] = False
            self.rgb[cpu][0].save(path+'/'+str(ep)+'.gif',
                                  save_all=True,
                                  append_images=self.rgb[cpu][1:],
                                  optimize=False,
                                  duration=50,
                                  loop=1)
            self.rgb[cpu] = []

        # Prepare next rendering step
        if (((ep+1)%self.render_every == 0) and (ep != 0)):
            self.render[cpu] = True
