## `cartpole-v0` (discrete)

The agent learns to balance a pole fixed to a moving cart, using discrete lateral movements of the cart.

<p align="center">
  <img width="300" alt="" src="bad.gif">
  <img width="300" alt="" src="good.gif">
</p>

Here is a resolution with PPO, using buffer-based training:

<p align="center">
  <img width="700" alt="" src="ppo_buffer.png">
</p>

and using episode-based training:

<p align="center">
  <img width="700" alt="" src="ppo_episode.png">
</p>

Below is a resolution of a continuous version of the cartpole environment, using buffer-based PPO:

<p align="center">
  <img width="700" alt="" src="ppo_buffer_continuous.png">
</p>
