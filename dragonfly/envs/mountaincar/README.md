## `mountaincar-v0` (discrete)

The agent learns to run a car from the bottom of a valley to the top of a hill. This problem has sparse reward (i.e. it is 0 everywhere, except when the cart reaches the final goal).

<p align="center">
  <img width="300" alt="" src="bad.gif">
  <img width="300" alt="" src="good.gif">
</p>

Here is a resolution with buffer-based ppo:

<p align="center">
  <img width="700" alt="" src="ppo_buffer.png">
</p>
