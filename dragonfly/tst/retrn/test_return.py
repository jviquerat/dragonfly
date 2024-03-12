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

    #########################
    # Test discounted return without bootstrap
    print("Disc. return, no norm, no bootstrap, no terminal")

    # Read json file and declare return
    reader = json_parser()
    reader.read("dragonfly/tst/retrn/full.json")
    retrn = full(reader.pms.retrn)

    # Compute returns
    trm      = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
    ret_ref  = np.array([4.90099501, 3.940399, 2.9701, 1.99, 1.0])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test discounted return with terminal state
    print("Disc. return, no norm, no bootstrap, with terminal")

    # Compute returns
    trm      = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
    ret_ref  = np.array([ 2.9701, 1.99, 1.0, 1.99, 1.0])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test discounted return with bootstrap
    print("Disc. return, no norm, with bootstrap, with terminal")

    # Compute returns
    trm      = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
    ret_ref  = np.array([3.940399, 2.9701, 1.99, 1.99, 1.0])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test discounted return with normalization
    print("Disc. return, normalized, no bootstrap, no terminal")

    # Modify retrn object
    retrn.ret_norm = True

    # Compute returns
    # avg is 2.960298802
    # std is 1.3792204603784504
    trm      = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
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
    print("Advantage, no norm, no bootstrap, no terminal")

    # Read json file and declare return
    reader.read("dragonfly/tst/retrn/advantage.json")
    retrn = advantage(reader.pms.retrn)

    # Compute returns
    trm      = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
    ret_ref  = np.array([0.90099501, 0.940399, 0.9701, 0.99, 0.5])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test advantage with terminal state
    print("Advantage, no norm, no bootstrap, with terminal")

    # Compute returns
    trm      = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
    ret_ref  = np.array([-1.0299, -1.01, -1.0, 0.99, 0.5])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test advantage with bootstrap
    print("Advantage, no norm, with bootstrap, with terminal")

    # Compute returns
    trm      = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
    ret_ref  = np.array([-0.059601, -0.0299, -0.01, 0.99, 0.5])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test vanilla GAE
    print("GAE, no norm, no bootstrap, no terminal")

    # Read json file and declare return
    reader.read("dragonfly/tst/retrn/gae.json")
    retrn = gae(reader.pms.retrn)

    # Compute returns
    trm      = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
    ret_ref  = np.array([1.01167994, 1.07367546, 1.12726805, 1.1721996, 0.698])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test GAE with terminal
    print("GAE, no norm, no bootstrap, with terminal")

    # Compute returns
    trm      = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
    ret_ref  = np.array([0.83624735, 0.89285441, 0.94089302, 0.9801, 0.5])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")

    #########################
    # Test GAE with bootstrap
    print("GAE, no norm, with bootstrap, with terminal")

    # Compute returns
    trm      = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    tgt, ret = retrn.compute(rwd, val, nxt, trm)
    ret_ref  = np.array([1.01167994, 1.07367546, 1.12726805, 1.1721996, 0.698])

    print("Reference return")
    print(ret_ref)
    print("Computed return")
    print(ret)

    assert(np.all(np.abs(ret-ret_ref)/np.abs(ret_ref + ret_eps) < 1.0e-8))
    print("")
