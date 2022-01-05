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
    buff      = glb_buff(n_cpu, obs_dim, act_dim)

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

    # Retrieve full buffer
    buff_obs, buff_act, buff_adv, buff_tgt = buff.get_buffers(n_buff, buff_size)
    assert(len(buff_obs)==n_buff)

    # Retrieve smaller buffer
    buff_obs, buff_act, buff_adv, buff_tgt = buff.get_buffers(n_buff-1, buff_size)
    assert(len(buff_obs)==n_buff-1)

    # Retrieve larger buffer
    buff_obs, buff_act, buff_adv, buff_tgt = buff.get_buffers(n_buff+2, buff_size)
    assert(len(buff_obs)==n_buff)
