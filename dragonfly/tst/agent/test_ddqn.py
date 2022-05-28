# Generic imports
import os
import shutil
import pytest

# Custom imports
from dragonfly.tst.tst             import *
from dragonfly.src.agent.ddqn      import *
from dragonfly.src.utils.json      import *
from dragonfly.src.utils.data      import *
from dragonfly.src.envs.par_envs   import *
from dragonfly.src.trainer.trainer import *

###############################################
### Test ddqn agent
def test_ddqn():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/agent/ddqn.json")

    # Intialize averager
    averager = data_avg(4, reader.pms.n_ep_max, reader.pms.n_avg)

    # Initialize trainer
    trainer = trainer_factory.create(reader.pms.trainer.style,
                                     env_name  = reader.pms.env_name,
                                     agent_pms = reader.pms.agent,
                                     path      = ".",
                                     n_cpu     = reader.pms.n_cpu,
                                     n_ep_max  = reader.pms.n_ep_max,
                                     pms       = reader.pms.trainer)

    print("Test ddqn")
    os.makedirs("0/", exist_ok=True)
    os.makedirs("1/", exist_ok=True)
    trainer.reset()
    trainer.loop(".", 0)
    averager.store("0/ddqn_0.dat", 0)
    trainer.reset()
    trainer.loop(".", 1)
    averager.store("1/ddqn_1.dat", 1)
    trainer.env.close()
    averager.average("ddqn_avg.dat")

    shutil.rmtree("0")
    shutil.rmtree("1")
    os.remove("ddqn_avg.dat")
    print("")
