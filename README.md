# ppo_discrete

This repository contains a PPO implementation for discrete environments.   
It does **not** strictly follows the original implementation from Schulman *et al.* (see https://arxiv.org/abs/1707.06347).  

The current implementation contains several variations compared to the original paper, such as:

- Advantage clipping : negative advantages can be clipped to 0
- Off-policy learning : the experience buffers from previous policies are kept and re-used several times before being discarded
- Bootstrapping : a specific care is given to the handling of the ```done``` signal, depending on its type (environment failure or episode termination)

Below are a few tests made on several classic ```gym``` environments.

## CartPole-v0

The agent learns to balance a pole fixed to a moving cart. The episode ends when the cart gets out of screen, or when the pole exceeds a certain angle. The goal is to balance the pole for 200 timesteps.

<p align="center">
  <img width="430" alt="" src="https://user-images.githubusercontent.com/44053700/83844966-45bfdc00-a708-11ea-98ee-5623162e1fe1.gif">
  <img width="430" alt="" src="https://user-images.githubusercontent.com/44053700/83844541-7a7f6380-a707-11ea-8a2c-148ca0c8f67b.gif">
</p>
