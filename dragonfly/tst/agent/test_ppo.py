# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst        import *
from dragonfly.src.agent.ppo  import *
from dragonfly.src.utils.json import *

###############################################
### Test ppo agent
def test_ppo():

    # Initial space
    print("")

    # Initialize json parser and read test json file
    reader_discrete = json_parser()
    reader_discrete.read("dragonfly/tst/agent/discrete.json")

    # Initialize discrete agent
    agent_discrete = ppo(1, 1, reader_discrete.pms)

    print("Test discrete agent")
    obs = [[1.0]]
    act = agent_discrete.get_actions(obs)
    assert((act == [0]) or (act == [1]))
    print("")

    # Initialize json parser and read test json file
    reader_continuous = json_parser()
    reader_continuous.read("dragonfly/tst/agent/continuous.json")

    # Initialize continuous agent
    agent_continuous = ppo(1, 100, reader_continuous.pms)

    print("Test continuous agent")
    obs = [[1.0]]
    act = agent_continuous.get_actions(obs)
    assert(np.all(act < 1.0))
    assert(np.all(act >-1.0))
    print("")
