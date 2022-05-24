# Custom imports
from dragonfly.src.core.factory import *
from dragonfly.src.agent.ppo    import *
from dragonfly.src.agent.dqn    import *
from dragonfly.src.agent.ddqn   import *
from dragonfly.src.agent.td3    import *

# Declare factory
agent_factory = factory()

# Register agents
agent_factory.register("ppo",  ppo)
agent_factory.register("dqn",  dqn)
agent_factory.register("ddqn", ddqn)
agent_factory.register("td3",  td3)
