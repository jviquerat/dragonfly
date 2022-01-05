# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst        import *
from dragonfly.src.utils.buff import *

###############################################
### Test global buffer class
def test_glb_buff():

    # Initial space
    print("")

    #########################
    # Initialize buffer
    n_cpu     = 1
    obs_dim   = 3
    act_dim   = 2
    n_buff    = 5
    buff_size = 1
    btc_frac  = 0.5
    buff      = glb_buff(n_cpu,  obs_dim,   act_dim,
                         n_buff, buff_size, btc_frac)

    # Create fake buffers
    obs = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [1.0, 0.0, 1.0]])
    act = np.array([[1.0, 0.5],
                    [0.0, 1.0],
                    [0.5, 1.0],
                    [1.0, 1.0],
                    [0.5, 0.5]])
    adv = np.array([[0.1],
                    [0.5],
                    [0.3],
                    [0.2],
                    [0.1]])
    tgt = np.array([[0.8],
                    [0.3],
                    [0.2],
                    [0.1],
                    [0.1]])
    buff.store(obs, adv, tgt, act)

    print("Storing buffers")
    print("obs: ")
    print(obs)
    print("act: ")
    print(act)
    print("adv: ")
    print(adv)
    print("tgt: ")
    print(tgt)

    # Retrieve buffer
    buff_obs, buff_act, buff_adv, buff_tgt = buff.get_buff()

    # Retrieve first batch
    start, end, done = buff.get_indices()
    btc_obs          = obs[start:end]
    btc_act          = act[start:end]
    btc_adv          = adv[start:end]
    btc_tgt          = tgt[start:end]

    assert(len(btc_obs)==2)
    assert(len(btc_act)==2)
    assert(len(btc_adv)==2)
    assert(len(btc_tgt)==2)
    assert(done==False)

    # Retrieve second batch
    start, end, done = buff.get_indices()
    btc_obs          = obs[start:end]
    btc_act          = act[start:end]
    btc_adv          = adv[start:end]
    btc_tgt          = tgt[start:end]

    assert(len(btc_obs)==2)
    assert(len(btc_act)==2)
    assert(len(btc_adv)==2)
    assert(len(btc_tgt)==2)
    assert(done==False)

    # Retrieve third batch
    start, end, done = buff.get_indices()
    btc_obs          = obs[start:end]
    btc_act          = act[start:end]
    btc_adv          = adv[start:end]
    btc_tgt          = tgt[start:end]

    assert(len(btc_obs)==1)
    assert(len(btc_act)==1)
    assert(len(btc_adv)==1)
    assert(len(btc_tgt)==1)
    assert(done==True)
