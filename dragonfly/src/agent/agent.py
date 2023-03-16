# Custom imports
from dragonfly.src.core.factory import *
from dragonfly.src.agent.a2c    import *
from dragonfly.src.agent.ppo    import *
from dragonfly.src.agent.dqn    import *
from dragonfly.src.agent.ddpg   import *
from dragonfly.src.agent.td3    import *
from dragonfly.src.agent.sac    import *
from dragonfly.src.agent.pgmu   import *

# Declare factory
agent_factory = factory()

# Register agents
agent_factory.register("a2c",  a2c)
agent_factory.register("ppo",  ppo)
agent_factory.register("dqn",  dqn)
agent_factory.register("ddpg", ddpg)
agent_factory.register("td3",  td3)
agent_factory.register("sac",  sac)
agent_factory.register("pgmu", pgmu)
