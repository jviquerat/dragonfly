from gym.envs.registration import register

register(
    id='cartpole-continuous-v0',
    entry_point='dragonfly.custom.cartpole:CartPoleContinuous'
)

register(
    id='pendulum-deterministic-v0',
    entry_point='dragonfly.custom.pendulum:PendulumDeterministic',
)
