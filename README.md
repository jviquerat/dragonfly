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
  <img width="300" alt="" src="https://user-images.githubusercontent.com/44053700/116778687-ab7d2b00-aa73-11eb-9be7-788581c052a0.gif">
  <img width="300" alt="" src="https://user-images.githubusercontent.com/44053700/116778684-a4eeb380-aa73-11eb-95ac-79d760bbc843.gif">
</p>

<p align="center">
  <img width="800" alt="" src="https://user-images.githubusercontent.com/44053700/116778701-cd76ad80-aa73-11eb-850a-32b9e0961156.png">
</p>

## `lunarlander-v2` (discrete)

The agent learns to land on a landing pad using 4 discrete actions.

<p align="center">
  <img width="300" alt="" src="https://user-images.githubusercontent.com/44053700/90507551-5bb13a80-e156-11ea-9037-4745e7c531d0.gif">
  <img width="300" alt="" src="https://user-images.githubusercontent.com/44053700/90507558-5e139480-e156-11ea-83a2-7529d23daf17.gif">
</p>

<p align="center">
  <img width="800" alt="" src="https://user-images.githubusercontent.com/44053700/94101304-df360980-fe2f-11ea-8975-9250801e18f0.png">
</p>
