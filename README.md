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

Environments are expected to be available locally or present in the path. To train an agent on an environment, a `.json` case file is required (sample files are available for standard `gym` envs in `dragonfly/envs`). Once you have written the corresponding `<env_name>.json` file to configure your agent, just run:

```
dgf --train <env_name>.json
```

To evaluate a trained agent, you will need a trained agent saved with `tf` format, as well as a `.json` case file. Then, just run:

``` 
dgf --eval -net <net_file> -json <json_file>
```

In that case, the environment will just rely on the `done` signal to stop the evaluation. Alternatively, you can provide a `-steps <n>` option, that will override the `done` signal of the environment, and force its execution for `n` steps. Trained agents for standard `gym` environements are available in `dragonfly/envs`.

## Solved environments

Below are a few environment solved with the library.

| Environment | Description | Illustration |
| :--- | :--- | :---: |
| `CartPole-v0` | The basic `gym` environment. The agent learns to balance a pole fixed to a moving cart, using discrete lateral movements of the cart. See additional details <a href="dragonfly/save/cartpole/README.md">here</a>. | <img width="500" alt="gif" src="dragonfly/save/cartpole/good.gif"> |
| `Pendulum-v0` | The basic `gym` environment. The agent learns to balance a 1-bar pendulum vertically, using limited torque force. See additional details <a href="dragonfly/save/pendulum/README.md">here</a>.  | <img width="500" alt="gif" src="dragonfly/save/pendulum/good.gif"> |
| `Acrobot-v1` | The basic `gym` environment. The agent learns to swing a two-links system up to a certain height. See additional details <a href="dragonfly/save/acrobot/README.md">here</a>.  | <img width="500" alt="gif" src="dragonfly/save/acrobot/good.gif">
| `LunarLander-v2` | The basic `gym` environment. The agent learns to land on a landing pad using 4 discrete actions. See additional details <a href="dragonfly/save/lunarlander/README.md">here</a>. | <img width="500" alt="gif" src="dragonfly/save/lunarlander/good.gif"> |
