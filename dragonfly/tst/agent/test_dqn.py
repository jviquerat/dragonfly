# Generic imports
import os
import shutil
import pytest

# Custom imports
from dragonfly.tst.tst             import *
from dragonfly.src.agent.dqn       import *
from dragonfly.src.utils.json      import *
from dragonfly.src.utils.data      import *
from dragonfly.src.envs.par_envs   import *
from dragonfly.src.trainer.trainer import *

###############################################
### Test dqn agent
def test_dqn():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/agent/dqn.json")

    # Initialize environment
    env = par_envs(reader.pms.env_name, reader.pms.n_cpu, ".")

    # Initialize discrete agent
    agent = dqn(4, 1, reader.pms.n_cpu, reader.pms.agent)

    # Intialize averager
    averager = data_avg(4, reader.pms.n_ep_max, reader.pms.n_avg)

    # Initialize training style
    trainer = trainer_factory.create(reader.pms.trainer.style,
                                     obs_dim  = env.obs_dim,
                                     act_dim  = env.act_dim,
                                     pol_dim  = agent.pol_dim,
                                     n_cpu    = reader.pms.n_cpu,
                                     n_ep_max = reader.pms.n_ep_max,
                                     pms=reader.pms.trainer)

    print("Test dqn")
    os.makedirs("0/", exist_ok=True)
    os.makedirs("1/", exist_ok=True)
    trainer.reset()
    agent.reset()
    trainer.loop(".", 0, env, agent)
    averager.store("0/dqn_0.dat", 0)
    trainer.reset()
    agent.reset()
    trainer.loop(".", 1, env, agent)
    averager.store("1/dqn_1.dat", 1)
    env.close()
    averager.average("dqn_avg.dat")

    shutil.rmtree("0")
    shutil.rmtree("1")
    os.remove("dqn_avg.dat")
    print("")
