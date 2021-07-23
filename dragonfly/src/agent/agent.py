# Custom imports
from dragonfly.src.core.factory import *
from dragonfly.src.agent.ppo    import *

# Declare factory
agent_factory = factory()

# Register agents
agent_factory.register("ppo", ppo)
