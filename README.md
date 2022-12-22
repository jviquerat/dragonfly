# dragonfly

<p align="center">
  <img align="right" width="350" alt="logo" src="dragonfly/msc/logo.png">
</p>

![master badge](https://github.com/jviquerat/dragonfly/workflows/dragonfly/badge.svg?branch=master)

`dragonfly` is a small library for DRL. It follows a basic level of modularity based on a simple abstract factory to make new developments quick and easy.

## Installation and usage

Clone this repository and install it locally:

```
git clone git@github.com:jviquerat/dragonfly.git
cd dragonfly
pip install -e .
```

Environments are expected to be available locally or present in the path. To train an agent on an environment, a `.json` case file is required (sample files are available for standard `gym` envs in `dragonfly/env`). Once you have written the corresponding `<env_name>.json` file to configure your agent, just run:

```
dgf --train <json_file>
```

To evaluate a trained agent, you will need a trained agent saved with `tf` format, as well as a `.json` case file. Then, just run:

``` 
dgf --eval -net <net_file> -json <json_file> -steps <n_steps_control> -warmup <n_steps_warmup> <warmup_control_value>
```

In that case, the environment will just rely on the `done` signal to stop the evaluation. Alternatively, you can provide a `-steps <n>` option, that will override the `done` signal of the environment, and force its execution for `n` steps. Trained agents for standard `gym` environements are available in `dragonfly/env`.

## CFD environments

| Environment | Description | Illustration |
| :--- | :--- | :---: |
| `turek-v0` | A drag reduction problem exploiting two synthetic jets on a cylinder immersed in a flow at Re=100. See additional details <a href="dragonfly/env/turek/README.md">here</a>. | <img width="400" alt="gif" src="dragonfly/env/turek/good.gif"> |
| `shkadov-v0` | A control problem with multiple jets trying to damp instabilities on a falling liquid film. See additional details <a href="dragonfly/env/shkadov/README.md">here</a>. | <img width="400" alt="gif" src="dragonfly/env/shkadov/good.gif"> |

## Mujoco environments

| [**`Hopper-v4`**](dragonfly/env/hopper/README.md)       | [**`Ant-v4`**](dragonfly/env/ant/README.md)           | [**`Swimmer-v4`**](dragonfly/env/hopper/README.md)       |
| :-----------------------------------------------------: | :---------------------------------------------------: | :-----------------------------------------------------: |
| <img height="250" alt="gif" src="dragonfly/env/hopper/good.gif"> | <img height="250" alt="gif" src="dragonfly/env/ant/good.gif"> | <img height="250" alt="gif" src="dragonfly/env/swimmer/good.gif">                |

## Gym environments

| [**`Cartpole-v0`**](dragonfly/env/cartpole/README.md)   | [**`Pendulum-v0`**](dragonfly/env/pendulum/README.md)   | [**`Acrobot-v1`**](dragonfly/env/acrobot/README.md)       |
| :-----------------------------------------------------: | :-----------------------------------------------------: | :-----------------------------------------------------: |
| <img height="200" alt="gif" src="dragonfly/env/cartpole/good.gif">  | <img height="200" alt="gif" src="dragonfly/env/pendulum/good.gif">            | <img height="200" alt="gif" src="dragonfly/env/acrobot/good.gif">                |
| [**`LunarLanderContinuous-v2`**](dragonfly/env/lunarlandercontinuous/README.md) | [**`BipedalWalker-v3`**](dragonfly/env/bipedalwalker/README.md)   | [**`MountainCar-v0`**](dragonfly/env/mountaincar/README.md)       |
| <img height="200" alt="gif" src="dragonfly/env/lunarlandercontinuous/good.gif"> | <img height="200" alt="gif" src="dragonfly/env/bipedalwalker/good_td3.gif">            | <img height="200" alt="gif" src="dragonfly/env/mountaincar/good.gif">                |
