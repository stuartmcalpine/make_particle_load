import numpy as np
from mpi4py import MPI

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

def message(*args, end="\n"):
    if comm_rank == 0:
        print(*args, end=end)

def print_section_header(header_name):
    if comm_rank == 0:
        print("-" * (len(header_name) + 8))
        print(f"--- {header_name} ---")
        print("-" * (len(header_name) + 8))
