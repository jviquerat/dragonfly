# ppo_discrete

This repository contains a PPO implementation for discrete environments.   
It does **not** strictly follows the original implementation from Schulman *et al.* (see https://arxiv.org/abs/1707.06347).  

The current implementation contains several variations compared to the original paper, such as:

- Advantage clipping : negative advantages can be clipped to 0
- Off-policy learning : the experience buffers from previous policies are kept and re-used several times before being discarded
- Buffer-based updates : the network updates are done using fixed-size buffers, independently of the episode completions
- Bootstrapping : a specific care is given to the handling of the ```done``` signal, depending on its type (environment failure or episode termination), in the fashion of what is done in the [`Tensorforce`](https://github.com/tensorforce/tensorforce) library.

Below are a few tests made on several ```gym``` environments.

## CartPole-v0

The agent learns to balance a pole fixed to a moving cart. The episode ends when the cart gets out of screen, or when the pole exceeds a certain angle. The goal is to balance the pole for 200 timesteps. The graphs shown below are averaged over 5 runs. The agent usually learns to balance the pole perfectly within 50 to 100 episodes.

<p align="center">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/83844966-45bfdc00-a708-11ea-98ee-5623162e1fe1.gif">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/83844541-7a7f6380-a707-11ea-8a2c-148ca0c8f67b.gif">
</p>

<p align="center">
  <img width="800" alt="" src="https://user-images.githubusercontent.com/44053700/84269522-9b8ded00-ab29-11ea-8095-1fdd42daddb2.png">
</p>

## gym_lorenz:lorenz-oscillator-v0

This is a <a href="https://github.com/jviquerat/gym-lorenz">custom environment</a> exploiting the Lorenz attractor. The environment aims at maximizing the number of sign changes of the x component of the attractor, by applying a discrete control on the y EDO (please see <a href="The first environment aims at maximizing the number of sign changes of the x component. Reward is consistently 0, except when x sign changes, in which case it is +1. The control-less environments has a reward of 14. Below is a sample of controlled vs uncontrolled environment, processed with an in-house PPO code. As can be seen, the control significantly increases the number of sign changes, while constraining the trajectory."> link</a> for additional information). Reward is consistently 0, except when x sign changes, in which case it is +1. The control-less environments has a reward of 14. Below is a sample of controlled vs uncontrolled environment. As can be seen, the control significantly increases the number of sign changes, while constraining the trajectory.

<p align="center">
  <img width="900" alt="oscillator_2D" src="https://user-images.githubusercontent.com/44053700/90250978-4c23b000-de3d-11ea-9dca-1f5194ed4754.png">
</p>

<p align="center">
  <img width="300" alt="uncontrolled_3D" src="https://user-images.githubusercontent.com/44053700/90246972-f1d32100-de35-11ea-883f-7476d6082e4d.png"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img width="300" alt="controlled_3D" src="https://user-images.githubusercontent.com/44053700/90246970-f0095d80-de35-11ea-9726-332167676a1c.png">
</p>

<p align="center">
  <img width="800" alt="oscillator_2D" src="https://user-images.githubusercontent.com/44053700/93884737-907b5900-fce3-11ea-94b0-6868865dfbfc.png">
</p>

## LunarLander-v2

The agent learns to land on a landing pad using 4 discrete actions. The episode ends prematurely if the lander crashes. Maximal reward is obtained if the lander gets on the pad at zero speed. The graphs shown below are averaged over 5 runs.

<p align="center">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/90507551-5bb13a80-e156-11ea-9037-4745e7c531d0.gif">
  <img width="410" alt="" src="https://user-images.githubusercontent.com/44053700/90507558-5e139480-e156-11ea-83a2-7529d23daf17.gif">
</p>

<p align="center">
  <img width="800" alt="" src="https://user-images.githubusercontent.com/44053700/90507545-58b64a00-e156-11ea-90cb-633a40f5dd09.png">
</p>
