# Generic imports
import gym

# Custom imports
from dragonfly.src.utils.json     import *
from dragonfly.src.agent.agent    import *
from dragonfly.src.envs.par_envs  import *
from dragonfly.src.utils.renderer import *
from dragonfly.src.utils.prints   import *

# Evaluate agent
def evaluate(net_file, json_file, ns, nw, aw):

    # Initialize json parser and read parameters
    parser = json_parser()
    pms    = parser.read(json_file)

    # Load environment
    env = par_envs(1, ".", pms.env)

    # Set environment control tag to true if possible
    env.set_control()

    # Create renderer
    rnd = renderer(1, "rgb_array", 1)
    rnd.render = [True]

    # Initialize agent
    agent = agent_factory.create(pms.agent.type,
                                 obs_dim = env.obs_dim,
                                 act_dim = env.act_dim,
                                 n_cpu   = 1,
                                 size    = 10000,
                                 pms     = pms.agent)

    # Load network
    agent.load(net_file)

    # Specify termination
    term_ns = True
    term_dn = False

    if (ns == 0):
        term_ns = False
        term_dn = True

    # Reset
    n   = 0
    scr = 0.0
    obs = env.reset_all()

    # Specify warmup (unrolling without control)
    if (nw > 0):
        # Retrieve action type
        t  = env.get_action_type()
        if (t == "continuous"): act = [[float(aw)]]
        if (t == "discrete"):   act = [[int(aw)]]

        # Loop with neutral action
        for i in range(nw):
            obs, rwd, dne, trc = env.step(act)

    # Unroll
    while True:
        act                = agent.control(obs)
        obs, rwd, dne, trc = env.step(act)
        scr               += rwd[0]
        n                 += 1

        rnd.store(env)
        if (term_ns and n >= ns): break
        if (term_dn and dne):     break

    rnd.finish(".", "/", 0, 0)
    env.close()

    # Print
    new_line()
    spacer()
    print('Performed steps:  '+str(n))
    spacer()
    print('Cumulated reward: '+str(scr))

