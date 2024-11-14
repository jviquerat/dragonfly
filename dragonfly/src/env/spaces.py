# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.utils.prints   import liner, spacer
from dragonfly.src.utils.error    import error

###############################################
### A class holding informations about obs/act dimensions, shapes and transformations
class environment_spaces:
    def __init__(self, pms, spaces):

        # Constants
        self.inf_obs_value = 1.0e8

        # Default values
        self.act_norm       = True
        self.obs_norm       = True
        self.obs_clip       = False
        self.obs_noise      = False
        self.obs_stack      = 1
        self.obs_grayscale  = False
        self.obs_downscale  = 1
        self.obs_frameskip  = 1
        self.manual_obs_max = 1.0

        # Optional values in parameters
        if hasattr(pms, "act_norm"):      self.act_norm       = pms.act_norm
        if hasattr(pms, "obs_norm"):      self.obs_norm       = pms.obs_norm
        if hasattr(pms, "obs_clip"):      self.obs_clip       = pms.obs_clip
        if hasattr(pms, "obs_noise"):     self.obs_noise      = pms.obs_noise
        if hasattr(pms, "obs_stack"):     self.obs_stack      = pms.obs_stack
        if hasattr(pms, "obs_grayscale"): self.obs_grayscale  = pms.obs_grayscale
        if hasattr(pms, "obs_downscale"): self.obs_downscale  = pms.obs_downscale
        if hasattr(pms, "obs_frameskip"): self.obs_frameskip  = pms.obs_frameskip
        if hasattr(pms, "obs_max"):       self.manual_obs_max = pms.obs_max

        # Action space
        action_space = spaces[0]
        self.act_type = type(action_space).__name__
        if (self.act_type not in ["Discrete", "Box"]):
            error("environment_spaces", "__init__",
                  "Unknown action space: "+self.act_type)

        if (self.act_type == "Discrete"):
            self.act_dim  = int(action_space.n)
            self.act_norm = False
            self.act_min  = 1.0
            self.act_max  = 1.0

        if (self.act_type == "Box"):
            self.act_dim = int(action_space.shape[0])
            self.act_min = action_space.low
            self.act_max = action_space.high

        self.act_avg = 0.5*(self.act_max + self.act_min)
        self.act_rng = 0.5*(self.act_max - self.act_min)

        # Observation space
        obs_space = spaces[1]
        self.obs_type = type(obs_space).__name__
        if (self.obs_type not in ["Box"]):
            error("environment_spaces", "__init__",
                  "Unknown observation space: "+self.obs_type)

        if (self.obs_type == "Box"):
            self.obs_min  = obs_space.low
            self.obs_max  = obs_space.high

            # Retrieve natural obs shape
            self.natural_obs_shape = list(obs_space.shape)
            n_dims                 = len(self.natural_obs_shape)

            # Compute natural_obs_dim
            self.natural_obs_dim = 1
            for i in range(n_dims):
                self.natural_obs_dim *= self.natural_obs_shape[i]

            # Compute processed_obs_shape
            self.processed_obs_shape = self.natural_obs_shape.copy()
            for i in range(n_dims):
                self.processed_obs_shape[i] = self.processed_obs_shape[i]//self.obs_downscale

            # Second and third dimensions if image
            if (n_dims > 1):

                # Alpha channel dimension is mandatory
                if (n_dims == 2):
                    self.processed_obs_shape.append(1);
                    n_dims = 3

                # Possible grayscaling
                if self.obs_grayscale: self.processed_obs_shape[2] = 1

            # Compute processed_obs_dim
            self.processed_obs_dim = 1
            for i in range(n_dims):
                self.processed_obs_dim *= self.processed_obs_shape[i]

            # Handle possible observation stacking
            self.true_obs_shape = self.processed_obs_shape.copy()
            if (n_dims == 1): self.true_obs_shape[0] *= self.obs_stack
            if (n_dims == 3): self.true_obs_shape[2] *= self.obs_stack

            self.true_obs_dim = 1
            for i in range(n_dims):
                self.true_obs_dim *= self.true_obs_shape[i]

        self.obs_min = np.where(self.obs_min < -self.inf_obs_value,
                                -self.manual_obs_max,
                                self.obs_min)
        self.obs_max = np.where(self.obs_max >  self.inf_obs_value,
                                self.manual_obs_max,
                                self.obs_max)
        self.obs_avg = 0.5*(self.obs_max + self.obs_min)
        self.obs_rng = 0.5*(self.obs_max - self.obs_min)

        # For pixel-based envs
        if (self.obs_grayscale):
            self.obs_avg = self.obs_avg[:,:,0]
            self.obs_rng = self.obs_rng[:,:,0]

        if (self.obs_downscale):
            s = self.obs_downscale

            if (len(self.obs_avg.shape) == 1):
                self.obs_avg = self.obs_avg[::s]
                self.obs_rng = self.obs_rng[::s]
            if (len(self.obs_avg.shape) == 2):
                self.obs_avg = self.obs_avg[::s,::s]
                self.obs_rng = self.obs_rng[::s,::s]

    # Process actions based on options
    def process_actions(self, act):

        if (self.act_norm):
            act = np.clip(act, -1.0, 1.0)
            for i in range(self.act_dim):
                act[i] = self.act_rng[i]*act[i] + self.act_avg[i]

        return act

    # Process observations
    def process_observations(self, obs):

        if (self.obs_grayscale):
            obs = self.grayscale_obs(obs)
        if (self.obs_downscale):
            obs = self.downscale_obs(obs)
        if (self.obs_clip):
            obs = self.clip_obs(obs)
        if (self.obs_norm):
            obs = self.norm_obs(obs)
        if (self.obs_noise > 0.0):
            obs = self.noise_obs(obs)

        return obs

    # Grayscale observations (for pixel-based envs)
    # Alpha channel is assumed to be last
    def grayscale_obs(self, obs):

        x = np.zeros((obs.shape[0], obs.shape[1]))
        x[:,:] = np.mean(obs, axis=2)

        return x

    # Downscale observations (for pixel-based envs)
    def downscale_obs(self, obs):

        if (len(obs.shape) == 1):
            x = obs[::self.obs_downscale]
        if (len(obs.shape) == 2):
            x = obs[::self.obs_downscale,::self.obs_downscale]

        return x

    # Clip observations
    def clip_obs(self, obs):

        for i in range(self.processed_obs_dim):
            obs[i] = np.clip(obs[i], self.obs_min[i], self.obs_max[i])

        return obs

    # Normalize observations
    def norm_obs(self, obs):

        obs -= self.obs_avg
        obs /= self.obs_rng

        return obs

    # Add noise to observations
    def noise_obs(self, obs):

        noise = np.random.normal(0.0, self.obs_noise, self.processed_obs_dim)
        for i in range(self.processed_obs_dim):
            obs[i] += noise[i]

        return obs

    # Print space informations
    def print(self):

        liner()
        print("Problem dimensions")
        spacer()
        print("Action dim:        " + str(self.act_dim))
        spacer()
        print("Natural obs shape: " + str(self.natural_obs_shape))
        spacer()
        print("Natural obs dim:   " + str(self.natural_obs_dim))
        spacer()
        print("True obs shape:    " + str(self.true_obs_shape))
        spacer()
        print("True obs dim:      " + str(self.true_obs_dim))
