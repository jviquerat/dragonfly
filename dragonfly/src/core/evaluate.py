# Generic imports
import gym

# Custom imports
from dragonfly.src.utils.json       import json_parser
from dragonfly.src.agent.agent      import agent_factory
from dragonfly.src.env.environments import environments
from dragonfly.src.utils.renderer   import renderer
from dragonfly.src.utils.prints     import new_line, spacer

# Evaluate agent
def evaluate(net_folder, json_file, ns, nw, aw):

    # Initialize json parser and read parameters
    parser = json_parser()
    pms    = parser.read(json_file)

    # Load environment
    env = environments(".", pms.env)

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
                                 obs_dim = env.obs_dim,
                                 act_dim = env.act_dim,
                                 n_cpu   = 1,
                                 size    = 10000,
                                 pms     = pms.agent)

    # Load network
    agent.load_policy(net_folder)

    # import numpy as np
    # w = agent.p.net.net[0].get_weights()[0]
    # b = agent.p.net.net[0].get_weights()[1]
    # w = np.mean(w, axis=1)
    # print(w.shape)
    # np.savetxt('w_ppo', w)
    # np.savetxt('b_ppo', b)

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
        scr += rwd[0]
        n   += 1

        rnd.store(env)
        if (term_ns and n >= ns): break
        if (term_dn and dne):     break

    import numpy as np
    x = np.reshape(np.arange(0, 1000, 1, dtype=int), (-1,1))
    xp = np.reshape(np.arange(0, 500, 1, dtype=int), (-1,1)) 
    y = np.ones((1000,1))

    mean_original = np.reshape(np.mean(agent.obs_original, axis=1), (-1,1))
    std_original  = np.reshape(np.std(agent.obs_original, axis=1), (-1,1))
    original = np.hstack((x, mean_original, std_original))

    mean_pca = np.reshape(np.mean(agent.obs_pca, axis=1), (-1,1))
    std_pca  = np.reshape(np.std(agent.obs_pca, axis=1), (-1,1))
    pca = np.hstack((xp, mean_pca, std_pca))

    xx = np.reshape(np.arange(0, 400, 1, dtype=int), (-1,1))
    pca_2d = np.reshape(np.transpose(agent.obs_pca[0:2,:]), (-1,2))
    pca_2d = np.hstack((xx, pca_2d))

    np.savetxt('original', original, fmt='%10.5f', delimiter=',')
    np.savetxt('pca', pca, fmt='%10.5f', delimiter=',')
    np.savetxt('pca_2d', pca_2d, fmt='%10.5f')



    #w_x_mean_original = np.multiply(mean_original, w)
    #np.savetxt('w_x_mean_original', w_x_mean_original)
    #print(mean_pca.shape, w.shape)
    #w_x_mean_pca = np.multiply(mean_pca[:,0], w)
    # np.savetxt('w_x_mean_pca', w_x_mean_pca)

    rnd.finish(".", 0, 0)
    env.close()

    # Print
    new_line()
    spacer()
    print('Performed steps:  '+str(n))
    spacer()
    print('Cumulated reward: '+str(scr))

