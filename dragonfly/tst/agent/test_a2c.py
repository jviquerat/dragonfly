# Generic imports
import os
import shutil
import pytest

# Custom imports
from dragonfly.tst.tst             import *
from dragonfly.src.agent.a2c       import *
from dragonfly.src.utils.json      import *
from dragonfly.src.utils.data      import *
from dragonfly.src.envs.par_envs   import *
from dragonfly.src.trainer.trainer import *

###############################################
### Test continuous a2c agent
def test_a2c_continuous():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/agent/a2c.json")

    # Intialize averager
    averager = data_avg(4, reader.pms.n_ep_max, reader.pms.n_avg)

    # Initialize trainer
    trainer = trainer_factory.create(reader.pms.trainer.style,
                                     env_pms   = reader.pms.env,
                                     agent_pms = reader.pms.agent,
                                     path      = ".",
                                     n_cpu     = reader.pms.n_cpu,
                                     n_ep_max  = reader.pms.n_ep_max,
                                     pms       = reader.pms.trainer)

    print("Test continuous a2c")
    os.makedirs("0/", exist_ok=True)
    os.makedirs("1/", exist_ok=True)
    trainer.reset()
    trainer.loop(".", 0)
    averager.store("0/0.dat", 0)
    trainer.reset()
    trainer.loop(".", 1)
    averager.store("1/1.dat", 1)
    trainer.env.close()
    averager.average("avg.dat")

    shutil.rmtree("0")
    shutil.rmtree("1")
    os.remove("avg.dat")
    print("")
