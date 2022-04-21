import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    pass

# Initialise MPI if necessary, or set up dummy values otherwise
comm = None
comm_rank = 0
comm_size = 1


def init_mpi(use_mpi):

    global using_mpi
    global comm
    global comm_rank
    global comm_size

    if use_mpi == 1:
        comm = MPI.COMM_WORLD
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        if comm_rank == 0:
            print("\nStarting on %d MPI ranks..." % comm_size)
    else:
        print("\nRunning without MPI, using one core only...")
