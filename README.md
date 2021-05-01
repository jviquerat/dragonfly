# dragonfly

Dragonfly is a buffer-based DRL research code. It follows a basic level of modularity to make new developments quick and easy. Below are a few tests made on several ```gym``` environments, using discrete and continuous PPO:

## CartPole-v0

The agent learns to balance a pole fixed to a moving cart. The episode ends when the cart gets out of screen, or when the pole exceeds a certain angle. The goal is to balance the pole for 200 timesteps. The graphs shown below are averaged over 5 runs. The agent usually learns to balance the pole perfectly within 50 to 100 episodes.

<p align="center">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/83844966-45bfdc00-a708-11ea-98ee-5623162e1fe1.gif">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/83844541-7a7f6380-a707-11ea-8a2c-148ca0c8f67b.gif">
</p>

<p align="center">
  <img width="800" alt="" src="https://user-images.githubusercontent.com/44053700/84269522-9b8ded00-ab29-11ea-8095-1fdd42daddb2.png">
</p>

## Pendulum-v0

The agent learns to balance a 1-bar pendulum vertically, using limited torque force.

<p align="center">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/116778687-ab7d2b00-aa73-11eb-9be7-788581c052a0.gif">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/116778684-a4eeb380-aa73-11eb-95ac-79d760bbc843.gif">
</p>

<p align="center">
  <img width="800" alt="" src="https://user-images.githubusercontent.com/44053700/116778701-cd76ad80-aa73-11eb-850a-32b9e0961156.png">
</p>

## LunarLander-v2

The agent learns to land on a landing pad using 4 discrete actions. The episode ends prematurely if the lander crashes. Maximal reward is obtained if the lander gets on the pad at zero speed. The graphs shown below are averaged over 5 runs.

<p align="center">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/90507551-5bb13a80-e156-11ea-9037-4745e7c531d0.gif">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/90507558-5e139480-e156-11ea-83a2-7529d23daf17.gif">
</p>

<p align="center">
  <img width="800" alt="" src="https://user-images.githubusercontent.com/44053700/94101304-df360980-fe2f-11ea-8975-9250801e18f0.png">
</p>
