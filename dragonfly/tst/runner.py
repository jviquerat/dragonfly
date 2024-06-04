# Generic imports
import os
import shutil

# Custom imports
from dragonfly.src.core.constants   import *
from dragonfly.src.core.paths       import *
from dragonfly.src.utils.json       import *
from dragonfly.src.utils.data       import *
from dragonfly.src.env.environments import *
from dragonfly.src.trainer.trainer  import *

###############################################
### Generic runner used in agent and trainer tests
def runner(json_file, agent_type):

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read(json_file)

    # Intialize averager
    averager = data_avg(2, int(reader.pms.n_stp_max/step_report), reader.pms.n_avg)

    # Initialize trainer
    trainer = trainer_factory.create(reader.pms.trainer.style,
                                     env_pms   = reader.pms.env,
                                     agent_pms = reader.pms.agent,
                                     n_stp_max = reader.pms.n_stp_max,
                                     pms       = reader.pms.trainer)

    print("Test " + agent_type)

    for i in range(2):
        paths.run = paths.results + '/' + str(i)
        os.makedirs(paths.run, exist_ok=True)
        trainer.reset()
        trainer.loop()
        trainer.agent.save_policy(paths.run)
        filename = paths.run + '/data.dat'
        averager.store(filename, i)

    filename = paths.run + '/data.dat'
    averager.store(filename, 1)

    # Test of the `control` method
    trainer.reset()
    obs = trainer.env.reset_all()
    trainer.agent.load_policy(paths.run)
    act = trainer.agent.control(obs)
    trainer.env.step(act)
    #############################

    trainer.env.close()
    averager.average("avg.dat")

    shutil.rmtree("0")
    shutil.rmtree("1")
    os.remove("avg.dat")
    print("")
