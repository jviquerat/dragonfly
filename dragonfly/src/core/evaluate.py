# Generic imports
import gym

# Custom imports
from dragonfly.src.utils.json     import *
from dragonfly.src.agent.agent    import *
from dragonfly.src.envs.par_envs  import *
from dragonfly.src.utils.renderer import *
from dragonfly.src.utils.prints   import *

# Evaluate agent
def evaluate(net_file, json_file, ns):

    # Initialize json parser and read parameters
    parser = json_parser()
    pms    = parser.read(json_file)

    # Load environment
    env = par_envs(pms.env_name, 1, ".")
    env.set_cpus()

    # Create renderer
    rnd = renderer(1, 1)
    rnd.render = [True]

    # Create agent
    agent = agent_factory.create(pms.agent.type,
                                 obs_dim = env.obs_dim,
                                 act_dim = env.act_dim,
                                 n_cpu   = 1,
                                 pms     = pms.agent)

    # Load network
    agent.load(net_file)

    # Specify termination
    term_ns = True
    term_dn = False

    if (ns == -1):
        term_ns = False
        term_dn = True

    # Unroll
    n   = 0
    crd = 0.0
    obs = env.reset_all()
    while True:
        act           = agent.control(obs)
        obs, rwd, dne = env.step(act)
        crd          += rwd[0]
        n            += 1

        env.render(rnd.render)
        rnd.store(env.render(rnd.render))
        if (term_ns and n >= ns): break
        if (term_dn and dne):     break

    rnd.finish(".", 0, 0)
    env.close()

    # Print
    new_line()
    spacer()
    print('Performed steps:  '+str(n))
    spacer()
    print('Cumulated reward: '+str(crd))

