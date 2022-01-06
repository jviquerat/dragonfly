# Generic imports
import os
import pytest

# Custom imports
from dragonfly.tst.tst             import *
from dragonfly.src.agent.ppo       import *
from dragonfly.src.utils.json      import *
from dragonfly.src.utils.data      import *
from dragonfly.src.envs.par_envs   import *
from dragonfly.src.trainer.trainer import *

###############################################
### Test buffer-based training
def test_buffer_based():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/trainer/buffer_based.json")

    # Initialize environment
    env = par_envs(reader.pms.env_name, reader.pms.n_cpu, ".")

    # Initialize discrete agent
    agent = ppo(4, 1, reader.pms.n_cpu, reader.pms.agent)

    # Intialize averager
    averager = data_avg(4, reader.pms.n_ep_max, reader.pms.n_avg)

    # Initialize training style
    trainer = trainer_factory.create(reader.pms.trainer.style,
                                     obs_dim     = env.obs_dim,
                                     act_dim     = env.act_dim,
                                     pol_act_dim = agent.pol_act_dim,
                                     n_cpu       = reader.pms.n_cpu,
                                     n_ep_max    = reader.pms.n_ep_max,
                                     pms=reader.pms.trainer)

    print("Test buffer-based trainer")
    trainer.reset()
    agent.reset()
    env.set_cpus()
    trainer.loop(".", 0, env, agent)
    averager.store("ppo_0.dat", 0)
    trainer.reset()
    agent.reset()
    env.set_cpus()
    trainer.loop(".", 1, env, agent)
    averager.store("ppo_1.dat", 1)
    env.close()
    averager.average("ppo_avg.dat")

    os.remove("ppo_0.dat")
    os.remove("ppo_1.dat")
    os.remove("ppo_avg.dat")
    print("")
