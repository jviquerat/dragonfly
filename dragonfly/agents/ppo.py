# Custom imports
from dragonfly.agents.agent   import *

###############################################
### PPO agent
class ppo(agent):
    def __init__(self, act_dim, obs_dim, pms):
        super().__init__(act_dim, obs_dim, pms)

        # Initialize from arguments
        self.name         = 'ppo'
        self.n_vars       = 9
        self.pol_clip     = pms.pol_clip

    # Training
    def train(self):

        # Save actor weights
        self.actor.save_weights()

        # Train actor and critic
        for epoch in range(self.n_epochs):

            # Retrieve data
            obs, act, adv, tgt = self.glb_buff.get_buff()
            done               = False

            # Visit all available history
            while not done:
                start, end, done = self.glb_buff.get_indices()
                btc_obs          = obs[start:end]
                btc_act          = act[start:end]
                btc_adv          = adv[start:end]
                btc_tgt          = tgt[start:end]

                self.train_actor (btc_obs, btc_adv, btc_act)
                self.train_critic(btc_obs, btc_tgt, end - start)

        # Update old networks
        self.actor.set_weights()

    ################################
    ### Actor/critic wrappings
    ################################

    # Training function for actor
    def train_actor(self, obs, adv, act):

        outputs          = self.actor.train(obs, adv, act,
                                            self.pol_clip,
                                            self.entropy_coef)
        self.actor_loss  = outputs[0]
        self.kl_div      = outputs[1]
        self.actor_gnorm = outputs[2]
        self.entropy     = outputs[3]

    # Training function for critic
    def train_critic(self, obs, tgt, size):

        outputs           = self.critic.train(obs, tgt, size)
        self.critic_loss  = outputs[0]
        self.critic_gnorm = outputs[1]

