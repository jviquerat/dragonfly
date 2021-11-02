# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst             import *
from dragonfly.src.agent.ppo       import *
from dragonfly.src.utils.json      import *
from dragonfly.src.utils.data      import *
from dragonfly.src.envs.par_envs   import *
from dragonfly.src.trainer.trainer import *

###############################################
### Test continuous ppo agent
def test_ppo_continuous():

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/agent/continuous.json")

    # Initialize environment
    env = par_envs(reader.pms.env_name, reader.pms.n_cpu, ".")

    # Initialize discrete agent
    agent = ppo(3, 1, reader.pms)

    # Intialize averager
    averager = data_avg(agent.n_vars, reader.pms.n_ep, reader.pms.n_avg)

    # Initialize training style
    trainer = trainer_factory.create(reader.pms.trainer.style,
                                     obs_dim     = env.obs_dim,
                                     act_dim     = env.act_dim,
                                     pol_act_dim = agent.pol_act_dim,
                                     pms=reader.pms)

    print("Test continuous agent")
    agent.reset()
    env.set_cpus()
    trainer.train(".", 0, env, agent)
    averager.store("ppo_0.dat", 0)
    agent.reset()
    env.set_cpus()
    trainer.train(".", 1, env, agent)
    averager.store("ppo_1.dat", 1)
    env.close()
    averager.average("ppo_avg.dat")
    print("")
