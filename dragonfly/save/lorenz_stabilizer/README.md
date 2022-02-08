## ```lorenz-stabilizer-discrete-v0```

The second environment aims at minimizing the number of sign changes of the `x` component. Reward is consistently 1 for each step with negative x, and consistently -1 for each step with positive x (hence this is a dense reward environment). The control-less environments has a reward of -356, while the current controlled example has a cumulated reward of 620. You can find more about this environment on <a href="https://github.com/jviquerat/custom_gym_envs">this repository</a>.

<p align="center">
  <img width="900" alt="oscillator_2D" src="lorenz_compare.png">
</p>

<p align="center">
  <img width="900" alt="" src="ppo.png">
</p>

<p align="center">
  <img width="500" alt="oscillator_2D" src="good.gif">
</p>
