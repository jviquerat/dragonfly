# Custom imports
from dragonfly.src.core.factory   import factory
from dragonfly.src.agent.a2c      import a2c
from dragonfly.src.agent.ppo      import ppo
from dragonfly.src.agent.ppo_srl  import ppo_srl
from dragonfly.src.agent.dqn      import dqn
from dragonfly.src.agent.ddpg     import ddpg
from dragonfly.src.agent.td3      import td3
from dragonfly.src.agent.sac      import sac
from dragonfly.src.agent.sac_auto import sac_auto

# Declare factory
agent_factory = factory()

# Register agents
agent_factory.register("a2c",      a2c)
agent_factory.register("ppo",      ppo)
agent_factory.register("ppo-srl",  ppo_srl)
agent_factory.register("dqn",      dqn)
agent_factory.register("ddpg",     ddpg)
agent_factory.register("td3",      td3)
agent_factory.register("sac",      sac)
agent_factory.register("sac_auto", sac_auto)
