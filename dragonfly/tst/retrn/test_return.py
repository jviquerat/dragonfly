# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst             import *
from dragonfly.src.retrn.full      import *
from dragonfly.src.retrn.advantage import *
from dragonfly.src.retrn.gae       import *
from dragonfly.src.utils.json      import *

###############################################
### Test return classes
def test_return():

    # Initial space
    print("")

    # Initialize vectors for tests
    size = 5
    rwd  = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    val  = np.array([4.0, 3.0, 2.0, 1.0, 0.5])
    nxt  = np.array([3.0, 2.0, 1.0, 0.5, 0.2])
    trm  = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    bts  = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    #########################
    # Test discounted return without bootstrap
    print("Disc. return, no norm, no clip, no bootstrap, no terminal")

    # Read json file and declare return
    reader = json_parser()
    reader.read("dragonfly/tst/retrn/full.json")
    retrn = full(reader.pms.retrn)

    # Compute returns
    tgt, ret = retrn.compute(rwd, val, nxt, trm, bts)
    ret_ref  = np.array([4.90099501, 3.940399, 2.9701, 1.99, 1.0])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test discounted return with terminal state
    print("Disc. return, no norm, no clip, no bootstrap, with terminal")

    # Modify trm vector
    trm[2] = 0.0

    # Compute returns
    tgt, ret = retrn.compute(rwd, val, nxt, trm, bts)
    ret_ref  = np.array([ 2.9701, 1.99, 1.0, 1.99, 1.0])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test discounted return with bootstrap
    print("Disc. return, no norm, no clip, with bootstrap, with terminal")

    # Modify bts vector
    bts[2] = 1.0

    # Compute returns
    tgt, ret = retrn.compute(rwd, val, nxt, trm, bts)
    ret_ref  = np.array([3.940399, 2.9701, 1.99, 1.99, 1.0])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test discounted return with normalization
    print("Disc. return, normalized, no clip, no bootstrap, no terminal")

    # Reset trm and bts vectors
    trm[2] = 1.0
    bts[2] = 0.0

    # Modify retrn object
    retrn.ret_norm = True

    # Compute returns
    # avg is 2.960298802
    # std is 1.3792204603784504
    tgt, ret = retrn.compute(rwd, val, nxt, trm, bts)
    ret_ref  = np.array([1.407096446,    0.7106189519,
                         0.007106331643,-0.7035124767,-1.421309253])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test vanilla advantage
    print("Advantage, no norm, no clip, no bootstrap, no terminal")

    # Read json file and declare return
    reader.read("dragonfly/tst/retrn/advantage.json")
    retrn = advantage(reader.pms.retrn)

    # Compute returns
    tgt, ret = retrn.compute(rwd, val, nxt, trm, bts)
    ret_ref  = np.array([4.90099501, 3.940399, 2.9701, 1.99, 1.])
    ret_ref -= val

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test vanilla GAE
    print("GAE, no norm, no clip, no bootstrap, no terminal")

    # Read json file and declare return
    reader.read("dragonfly/tst/retrn/gae.json")
    retrn = gae(reader.pms.retrn)

    # Compute returns
    tgt, ret = retrn.compute(rwd, val, nxt, trm, bts)
    ret_ref  = np.array([1.01167994, 1.07367546,
                         1.12726805, 1.1721996, 0.698])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")
