# ppo_discrete

This repository contains a PPO implementation for discrete environments.   
It does **not** strictly follows the original implementation from Schulman *et al.* (see https://arxiv.org/abs/1707.06347).  

The current implementation contains several variations compared to the original paper, such as:

- Advantage clipping : negative advantages can be clipped to 0
- Off-policy learning : the experience buffers from previous policies are kept and re-used several times before being discarded
- Buffer-based updates : the network updates are done using fixed-size buffers, independently of the episode completions
- Bootstrapping : a specific care is given to the handling of the ```done``` signal, depending on its type (environment failure or episode termination), in the fashion of what is done in the [`Tensorforce`](https://github.com/tensorforce/tensorforce) library.

Below are a few tests made on several classic ```gym``` environments.

## CartPole-v0

The agent learns to balance a pole fixed to a moving cart. The episode ends when the cart gets out of screen, or when the pole exceeds a certain angle. The goal is to balance the pole for 200 timesteps. The graphs shown below are averaged over 5 runs. The agent usually learns to balance the pole perfectly within 50 to 100 episodes.

<p align="center">
  <img width="430" alt="" src="https://user-images.githubusercontent.com/44053700/83844966-45bfdc00-a708-11ea-98ee-5623162e1fe1.gif">
  <img width="430" alt="" src="https://user-images.githubusercontent.com/44053700/83844541-7a7f6380-a707-11ea-8a2c-148ca0c8f67b.gif">
</p>

<p align="center">
  <img width="800" alt="" src="https://user-images.githubusercontent.com/44053700/84269522-9b8ded00-ab29-11ea-8095-1fdd42daddb2.png">
</p>
