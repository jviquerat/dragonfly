# Generic imports
import gymnasium as gym

# Custom imports
from dragonfly.src.utils.json      import json_parser
from dragonfly.src.agent.agent     import agent_factory
from dragonfly.src.env.environment import environment
from dragonfly.src.utils.renderer  import renderer
from dragonfly.src.utils.prints    import new_line, spacer

# Evaluate agent
def evaluate(net_folder, json_file, ns, nw, aw, eval_frequency):

    # Initialize json parser and read parameters
    parser = json_parser()
    pms    = parser.read(json_file)

    # Load environment
    env = environment(".", pms.env)

    # Set environment control tag to true if possible
    env.set_control()

    # Create renderer
    rnd_style = "rgb_array"
    if hasattr(pms.trainer, "rnd_style"):
        rnd_style = pms.trainer.rnd_style
    rnd = renderer(1, rnd_style, 1)
    rnd.render = [True]

    # Initialize agent
    agent = agent_factory.create(pms.agent.type,
                                 spaces   = env.spaces,
                                 n_cpu    = 1,
                                 mem_size = 10000,
                                 pms      = pms.agent)

    # Load network
    agent.load_policy(net_folder)

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
        if (t == "continuous"):
            act = []
            for a in aw: act.append(float(a))
            act = [act]
        if (t == "discrete"):
            act = []
            for a in aw: act.append(int(a))

        # Loop with neutral action
        for i in range(nw):
            obs, rwd, dne, trc = env.step(act)
            rnd.store(env)

    # Unroll
    while True:
        act                = agent.control(obs)
        obs, rwd, dne, trc = env.step(act)
        scr               += rwd[0]

        if (n%eval_frequency == 0): rnd.store(env)
        if (term_ns and n >= ns-1): break
        if (term_dn and dne):       break

        n  += 1

    rnd.store(env)
    rnd.finish(".", 0, 0)
    env.close()

    # Print
    new_line()
    spacer()
    print('Performed steps:  '+str(n))
    spacer()
    print('Cumulated reward: '+str(scr))

