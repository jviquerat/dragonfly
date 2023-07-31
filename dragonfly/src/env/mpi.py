import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize   = False
from mpi4py import MPI

###############################################
### A wrapper class for mpi
class dgf_mpi:
    def __init__(self):

        MPI.Init()

        self.comm = MPI.COMM_WORLD
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()

    def finalize(self):

        MPI.Finalize()
        exit(0)

if MPI.Is_initialized():
    pass
else:
    mpi = dgf_mpi()
