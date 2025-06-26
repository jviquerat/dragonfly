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
        self.inf_obs_value_ = 1.0e8

        # Default values
        self.act_norm_       = True
        self.obs_norm_       = True
        self.norm_type_      = "min_max"
        self.obs_clip_       = False
        self.obs_noise_      = False
        self.obs_stack_      = 1
        self.obs_grayscale_  = False
        self.obs_downscale_  = 1
        self.obs_frameskip_  = 1
        self.manual_obs_max_ = 1.0
        self.separable_      = False

        # Optional values in parameters
        if hasattr(pms, "act_norm"):      self.act_norm_       = pms.act_norm
        if hasattr(pms, "obs_norm"):      self.obs_norm_       = pms.obs_norm
        if hasattr(pms, "norm_type"):     self.norm_type_      = pms.norm_type
        if hasattr(pms, "obs_clip"):      self.obs_clip_       = pms.obs_clip
        if hasattr(pms, "obs_noise"):     self.obs_noise_      = pms.obs_noise
        if hasattr(pms, "obs_stack"):     self.obs_stack_      = pms.obs_stack
        if hasattr(pms, "obs_grayscale"): self.obs_grayscale_  = pms.obs_grayscale
        if hasattr(pms, "obs_downscale"): self.obs_downscale_  = pms.obs_downscale
        if hasattr(pms, "obs_frameskip"): self.obs_frameskip_  = pms.obs_frameskip
        if hasattr(pms, "obs_max"):       self.manual_obs_max_ = pms.obs_max
        if hasattr(pms, "separable"):     self.separable_      = pms.separable

        # Check norm_type_
        if (self.norm_type_ not in ["min_max", "rsnorm"]):
            error("environment_spaces", "__init__",
                  "Unknown norm type: "+self.norm_type_)

        # Action space
        action_space = spaces[0]
        self.act_type_ = type(action_space).__name__
        if (self.act_type_ not in ["Discrete", "Box"]):
            error("environment_spaces", "__init__",
                  "Unknown action space: "+self.act_type_)

        if (self.act_type_ == "Discrete"):
            self.natural_act_dim_ = int(action_space.n)
            self.store_act_dim_   = 1
            self.true_act_dim_    = int(action_space.n)
            self.act_norm_        = False
            self.act_min_         = 1.0
            self.act_max_         = 1.0

        if (self.act_type_ == "Box"):
            self.natural_act_dim_ = int(action_space.shape[0])
            self.store_act_dim_   = int(action_space.shape[0])
            self.true_act_dim_    = self.natural_act_dim_
            self.act_min_         = action_space.low
            self.act_max_         = action_space.high

            # Handle possible separable environment
            # XXX For now, we assume that the resulting action dimension is 1
            if (self.separable_):
                self.store_act_dim_   = 1
                self.true_act_dim_    = 1

        self.act_avg_ = 0.5*(self.act_max_ + self.act_min_)
        self.act_rng_ = 0.5*(self.act_max_ - self.act_min_)

        # Observation space
        obs_space = spaces[1]
        self.obs_type_ = type(obs_space).__name__
        if (self.obs_type_ not in ["Box"]):
            error("environment_spaces", "__init__",
                  "Unknown observation space: "+self.obs_type_)

        if (self.obs_type_ == "Box"):
            self.obs_min_  = obs_space.low
            self.obs_max_  = obs_space.high

            # Retrieve natural obs shape
            self.natural_obs_shape_ = list(obs_space.shape)
            n_dims                  = len(self.natural_obs_shape_)

            # Compute natural_obs_dim
            self.natural_obs_dim_ = 1
            for i in range(n_dims):
                self.natural_obs_dim_ *= self.natural_obs_shape_[i]

            # Compute processed_obs_shape
            self.processed_obs_shape_ = self.natural_obs_shape_.copy()
            for i in range(0, min(2,n_dims)):
                self.processed_obs_shape_[i] //= self.obs_downscale_

            # Second and third dimensions if image
            if (n_dims > 1):

                # Alpha channel dimension is mandatory
                if (n_dims == 2):
                    self.processed_obs_shape_.append(1);
                    n_dims = 3

                # Possible grayscaling
                if self.obs_grayscale_: self.processed_obs_shape_[2] = 1

            # Compute processed_obs_dim
            self.processed_obs_dim_ = 1
            for i in range(n_dims):
                self.processed_obs_dim_ *= self.processed_obs_shape_[i]

            # Handle possible observation stacking
            self.true_obs_shape_ = self.processed_obs_shape_.copy()
            if (n_dims == 1): self.true_obs_shape_[0] *= self.obs_stack_
            if (n_dims == 3): self.true_obs_shape_[2] *= self.obs_stack_

            self.true_obs_dim_ = 1
            for i in range(n_dims):
                self.true_obs_dim_ *= self.true_obs_shape_[i]

        # Obs min, max, avg and rng
        self.obs_min_ = np.where(self.obs_min_ < -self.inf_obs_value_,
                                -self.manual_obs_max_,
                                 self.obs_min_)
        self.obs_max_ = np.where(self.obs_max_ >  self.inf_obs_value_,
                                 self.manual_obs_max_,
                                 self.obs_max_)
        self.obs_avg_ = 0.5*(self.obs_max_ + self.obs_min_)
        self.obs_rng_ = 0.5*(self.obs_max_ - self.obs_min_)

        # Compute input_obs_dim and input_obs_shape
        self.input_obs_dim_   = self.true_obs_dim_
        self.input_obs_shape_ = self.true_obs_shape_.copy()

        # If obs normalization is rsnorm
        if (self.norm_type_ == "rsnorm"):
            self.obs_count_ = 0.0
            self.obs_mu_    = np.zeros(self.true_obs_dim_)
            self.obs_std_   = np.zeros(self.true_obs_dim_)

        # Grayscale for pixel-based envs
        if (self.obs_grayscale_):
            self.obs_avg_ = self.obs_avg_[:,:,0]
            self.obs_rng_ = self.obs_rng_[:,:,0]

        # Downscale for pixel-based envs
        if (self.obs_downscale_):
            s = self.obs_downscale_

            if (len(self.obs_avg_.shape) == 1):
                self.obs_avg_ = self.obs_avg_[::s]
                self.obs_rng_ = self.obs_rng_[::s]
            if (len(self.obs_avg_.shape) == 2):
                self.obs_avg_ = self.obs_avg_[::s,::s]
                self.obs_rng_ = self.obs_rng_[::s,::s]
            if (len(self.obs_avg_.shape) == 3):
                self.obs_avg_ = self.obs_avg_[::s,::s,:]
                self.obs_rng_ = self.obs_rng_[::s,::s,:]

    # Accessor
    def input_obs_dim(self):
        return self.input_obs_dim_

    # Accessor
    def input_obs_shape(self):
        return self.input_obs_shape_

    # Accessor
    def true_obs_dim(self):
        return self.true_obs_dim_

    # Accessor
    def true_obs_shape(self):
        return self.true_obs_shape_

    # Accessor
    def processed_obs_dim(self):
        return self.processed_obs_dim_

    # Accessor
    def obs_stack(self):
        return self.obs_stack_

    # Accessor
    def obs_frameskip(self):
        return self.obs_frameskip_

    # Accessor
    def natural_act_dim(self):
        return self.natural_act_dim_

    # Accessor
    def store_act_dim(self):
        return self.store_act_dim_

    # Accessor
    def true_act_dim(self):
        return self.true_act_dim_

    # Process actions based on options
    def process_actions(self, act):

        if (self.act_norm_):
            act = np.clip(act, -1.0, 1.0)
            for i in range(self.true_act_dim_):
                act[i] = self.act_rng_[i]*act[i] + self.act_avg_[i]

        return act

    # Process observations
    def process_observations(self, obs):

        if (obs.dtype.kind in ["u", "i"]):
            obs = obs.astype(np.float32)

        if (self.obs_grayscale_):
            obs = self.grayscale_obs(obs)
        if (self.obs_downscale_):
            obs = self.downscale_obs(obs)
        if (self.obs_clip_):
            obs = self.clip_obs(obs)
        if (self.obs_norm_):
            obs = self.norm_obs(obs)
        if (self.obs_noise_ > 0.0):
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
            return obs[::self.obs_downscale_]
        if (len(obs.shape) == 2):
            return obs[::self.obs_downscale_,::self.obs_downscale_]
        if (len(obs.shape) == 3):
            return obs[::self.obs_downscale_,::self.obs_downscale_,:]

    # Clip observations
    def clip_obs(self, obs):

        for i in range(self.processed_obs_dim_):
            obs[i] = np.clip(obs[i], self.obs_min_[i], self.obs_max_[i])

        return obs

    # Normalize observations
    def norm_obs(self, obs):

        if (self.norm_type_ == "min_max"):
            obs -= self.obs_avg_
            obs /= self.obs_rng_

        if (self.norm_type_ == "rsnorm"):
            self.obs_count_ += 1
            delta          = obs - self.obs_mu_
            self.obs_mu_  += (1.0/self.obs_count_)*delta
            self.obs_std_  = ((self.obs_count_-1.0)/self.obs_count_)*(self.obs_std_ + delta*delta/self.obs_count_)

            obs = (obs - self.obs_mu_)/np.sqrt(self.obs_std_ + 1.0e-8)

        return obs

    # Add noise to observations
    def noise_obs(self, obs):

        noise = np.random.normal(0.0, self.obs_noise_, self.processed_obs_dim_)
        for i in range(self.processed_obs_dim_):
            obs[i] += noise[i]

        return obs

    # Print space informations
    def print(self):

        liner()
        print("Problem dimensions")
        spacer()
        print("Natural act dim:   " + str(self.natural_act_dim_))
        spacer()
        print("True act dim:      " + str(self.true_act_dim_))
        spacer()
        print("Store act dim:     " + str(self.store_act_dim_))
        spacer()
        print("Natural obs shape: " + str(self.natural_obs_shape_))
        spacer()
        print("Natural obs dim:   " + str(self.natural_obs_dim_))
        spacer()
        print("True obs shape:    " + str(self.true_obs_shape_))
        spacer()
        print("True obs dim:      " + str(self.true_obs_dim_))
