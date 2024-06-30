# Custom imports
from dragonfly.src.policy.tfd import *
from dragonfly.src.policy.base import base_policy
import torch
import torch.nn as nn
import torch.distributions as dist

###############################################
### Categorical policy class (discrete)
### obs_dim : input  dimension
### act_dim : output dimension
### pms     : parameters
class categorical(base_policy):
    def __init__(self, obs_dim, act_dim, pms, target=False):

        # Fill structure
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.dim        = self.act_dim
        self.out_dim    = [self.dim]
        self.store_dim  = 1
        self.store_type = int
        self.target     = target

        # Define and init network
        if pms.network.heads.final[0] != "softmax":
            warning("categorical", "__init__",
                    "Chosen final activation for categorical policy is not softmax")

        # Init from base class
        super().__init__(pms)

    # Get actions
    def actions(self, obs):
        act, lgp = self.sample(obs)
        act = act.detach().numpy().reshape(-1)
        lgp = lgp.detach().numpy().reshape(-1)

        return act, lgp

    # Control (deterministic actions)
    def control(self, obs):
        probs = self.forward(obs)
        act = torch.argmax(probs[0], dim=-1)
        act = act.detach().numpy().reshape(-1)

        return act

    # Sample actions
    def sample(self, obs):
        # Generate pdf
        pdf = self.compute_pdf(obs)

        # Sample actions
        act = pdf.sample((1,))
        lgp = pdf.log_prob(act)

        return act, lgp

    # Compute pdf
    def compute_pdf(self, obs):
        probs = self.forward(obs)
        return dist.Categorical(probs=probs[0])

    # Network forward pass
    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.from_numpy(obs).to(torch.float32)
        else:
            obs_tensor = obs
        return self.net(obs_tensor)

    # Reshape actions
    def reshape_actions(self, act):
        return act.reshape(-1)

    # Random uniform actions for warmup
    def random_uniform(self, obs):
        n_cpu = obs.shape[0]
        act = torch.randint(0, self.act_dim, size=(n_cpu, 1))

        return act.reshape(-1).numpy()
