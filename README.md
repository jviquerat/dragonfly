# dragonfly

Dragonfly is a buffer-based DRL research code. It follows a basic level of modularity to make new developments quick and easy. Below are a few tests made on several `gym` environments, using discrete and continuous PPO:

## `cartpole-v0` (discrete)

The agent learns to balance a pole fixed to a moving cart, using discrete lateral movements of the cart.

<p align="center">
  <img width="300" alt="" src="dragonfly/save/cartpole/bad.gif">
  <img width="300" alt="" src="dragonfly/save/cartpole/good.gif">
</p>

<p align="center">
  <img width="700" alt="" src="dragonfly/save/cartpole/ppo.png">
</p>

## `pendulum-v0` (continuous)

The agent learns to balance a 1-bar pendulum vertically, using limited torque force.

<p align="center">
  <img width="300" alt="" src="dragonfly/save/pendulum/bad.gif">
  <img width="300" alt="" src="dragonfly/save/pendulum/good.gif">
</p>

<p align="center">
  <img width="700" alt="" src="dragonfly/save/pendulum/ppo.png">
</p>

## `lunarlander-v2` (discrete)

The agent learns to land on a landing pad using 4 discrete actions.

<p align="center">
  <img width="300" alt="" src="dragonfly/save/lunarlander/bad.gif">
  <img width="300" alt="" src="dragonfly/save/lunarlander/good.gif">
</p>

<p align="center">
  <img width="700" alt="" src="dragonfly/save/lunarlander/ppo.png">
</p>
