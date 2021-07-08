# Custom imports
from dragonfly.core.factory import *
from dragonfly.agent.ppo    import *

# Declare factory
agent_factory = factory()

# Register agents
agent_factory.register("ppo", ppo)
