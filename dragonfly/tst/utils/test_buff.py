# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst        import *
from dragonfly.src.utils.buff import *

###############################################
### Test buffer class
def test_buff():

    # Initial space
    print("")

    #########################
    # Initialize buffer
    # First test is done with 1 cpu
    n_cpu = 1
    names = ["obs", "act", "rwd"]
    dims  = [3, 2, 1]
    loc_buff = buff(n_cpu, names, dims)

    # Create buffers to append
    obs = [[1.0, 0.0, 0.5]]
    act = [[0.1, 0.2]]
    rwd = [[1.0]]

    # Append to par_buff several times
    loc_buff.store(["obs", "act", "rwd"], [obs, act, rwd])
    loc_buff.store(["act", "rwd", "obs"], [act, rwd, obs])

    # Serialize and check content
    data = loc_buff.serialize(["obs"])
    sobs = data["obs"]
    assert(sobs[0]==obs[0]).all()
    assert(sobs[1]==obs[0]).all()

    print("Serialized obs buff after 2 append operations, on 1 cpu")
    print(sobs)
    print("")

    # Test size and length
    assert(loc_buff.length() == 2)
    assert(loc_buff.size()   == 2)

    # Test reset
    loc_buff.reset()
    assert(loc_buff.length() == 0)

    #########################
    # Same test with 2 cpus
    n_cpu = 2
    names = ["obs", "act", "rwd"]
    dims  = [3, 2, 1]
    loc_buff = buff(n_cpu, names, dims)

    # Create buffers to append
    obs = [[1.0, 0.0, 0.5],
           [2.0, 0.0, 1.0]]
    act = [[0.1, 0.2],
           [0.2, 0.4]]
    rwd = [[1.0],
           [2.0]]

    # Append to par_buff several times
    loc_buff.store(["obs", "act", "rwd"], [obs, act, rwd])
    loc_buff.store(["act", "rwd", "obs"], [act, rwd, obs])

    # Serialize and check content
    data = loc_buff.serialize(["obs"])
    sobs = data["obs"]
    assert(sobs[0]==obs[0]).all()
    assert(sobs[1]==obs[0]).all()
    assert(sobs[2]==obs[1]).all()
    assert(sobs[3]==obs[1]).all()

    print("Serialized obs buff after 2 append operations, on 2 cpus")
    print(sobs)
    print("")

    # Test size and length
    assert(loc_buff.length() == 2)
    assert(loc_buff.size()   == 4)
