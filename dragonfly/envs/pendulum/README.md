## `pendulum-v0` (continuous)

The agent learns to balance a 1-bar pendulum vertically, using limited torque force.

<p align="center">
  <img width="300" alt="" src="bad.gif">
  <img width="300" alt="" src="good.gif">
</p>

Solved with PPO:

<p align="center">
  <img width="700" alt="" src="ppo_buffer.png">
</p>

Yet, PPO is significantly outperformed by offline algorithms such as DDPG (top) and TD3 (bottom):

<p align="center">
  <img width="700" alt="" src="ddpg.png">
</p>

<p align="center">
  <img width="700" alt="" src="td3.png">
</p>
