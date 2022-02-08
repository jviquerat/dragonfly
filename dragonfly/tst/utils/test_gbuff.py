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
    names = ["obs", "act", "adv", "tgt", "lgp"]
    dims  = [3, 2, 1, 1, 1]
    n     = 5
    n_tot = 9
    buff  = gbuff(n_tot, names, dims)

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
    lgp = np.array([[0.1],
                    [0.2],
                    [0.3],
                    [0.1],
                    [0.1]])
    buff.store(names, [obs, act, adv, tgt, lgp])

    print("Storing buffers")
    print("obs: ")
    print(obs)
    print("act: ")
    print(act)
    print("adv: ")
    print(adv)
    print("tgt: ")
    print(tgt)
    print("lgp: ")
    print(lgp)

    # Test length
    assert(buff.length() == n)

    # Retrieve full buffer
    data = buff.get_buffers(names, n, shuffle=False)
    assert(len(data["obs"])==n)
    ob, ac, ad, tg, lg = (data[name] for name in names)
    assert(ob.numpy()==obs).all()
    assert(ac.numpy()==act).all()

    # Retrieve smaller buffer
    data = buff.get_buffers(names, n-1)
    assert(len(data["obs"])==(n-1))

    # Retrieve larger buffer
    data = buff.get_buffers(names, n+2)
    assert(len(data["obs"])==n)

    # Retrieve only one field
    data = buff.get_buffers(["adv"], n)
    assert(len(data["adv"])==n)

    # Re-store buffers
    buff.store(names, [obs, act, adv, tgt, lgp])
    assert(buff.length() == n_tot)

    # Retrieve full buffer again
    data = buff.get_buffers(names, n, shuffle=False)
    assert(len(data["obs"])==n)
    ob, ac, ad, tg, lg = (data[name] for name in names)
    assert(ob.numpy()==obs).all()
    assert(ac.numpy()==act).all()
