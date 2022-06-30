## `pendulum-v0` (continuous)

The agent learns to balance a 1-bar pendulum vertically, using limited torque force.

<p align="center">
  <img width="300" alt="" src="bad.gif">
  <img width="300" alt="" src="good.gif">
</p>

In this environment, we noticed that re-using previous buffers significantly reduces the variability during the learning process.

<p align="center">
  <img width="700" alt="" src="ppo_buffer.png">
</p>
