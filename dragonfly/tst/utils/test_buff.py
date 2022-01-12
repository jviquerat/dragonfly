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

    # Append to buffer
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

    print("Serialized obs buff after 2 append operations")
    print(sobs)
    print("")

    # #########################
    # # Same test with 2 cpus
    # n_cpu   = 2
    # obs_dim = 3
    # buff    = par_buff(n_cpu, obs_dim)

    # # Generate a vector to fill buffer
    # vec = np.array([])
    # v1  = 1.0*np.ones((obs_dim,1))
    # v2  = 2.0*np.ones((obs_dim,1))
    # vec = np.append(vec, v1)
    # vec = np.append(vec, v2)
    # vec = np.reshape(vec, (-1,obs_dim))

    # print("Vector to append to par_buff")
    # print(vec)
    # print("")

    # # Append to par_buff several times
    # buff.append(vec)
    # buff.append(vec)

    # print("par_buff after 2 append operations")
    # print(buff.buff)
    # print("")

    # # Serialize
    # arr = buff.serialize()

    # print("Serialized par_buff")
    # print(arr)
    # print("")

    # tst_arr = np.array([[1, 1, 1],
    #                     [1, 1, 1],
    #                     [2, 2, 2],
    #                     [2, 2, 2]])

    # assert (arr==tst_arr).all()

    # #########################
    # # Same test with inputs of size 1
    # n_cpu   = 2
    # obs_dim = 1
    # buff    = par_buff(n_cpu, obs_dim)

    # # Generate a vector to fill buffer
    # #vec = 1.0*np.ones((obs_dim))

    # vec = np.array([])
    # v1  = 1.0*np.ones((obs_dim))
    # v2  = 2.0*np.ones((obs_dim))
    # vec = np.append(vec, v1)
    # vec = np.append(vec, v2)

    # print("Vector to append to par_buff")
    # print(vec)
    # print("")

    # # Append to par_buff several times
    # buff.append(vec)
    # buff.append(vec)

    # print("par_buff after 2 append operations")
    # print(buff.buff)
    # print("")

    # # Serialize
    # arr = buff.serialize()

    # print("Serialized par_buff")
    # print(arr)
    # print("")

    # tst_arr = np.array([[1],
    #                     [1],
    #                     [2],
    #                     [2]])

    # assert (arr==tst_arr).all()
