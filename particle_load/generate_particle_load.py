import argparse

import numpy as np

import mympi
from HighResolutionRegion import HighResolutionRegion
from ic_gen_functions import compute_fft_stats
from LowResolutionRegion import LowResolutionRegion
from make_param_files import *
from ParticleLoadParams import ParticleLoadParams
from populate_particles import populate_particles
from ZoomRegionMask import ZoomRegionMask

# Command line args.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--make_ic_gen_param_files", help="Make ic_gen files.", action="store_true"
)
parser.add_argument("--param_file", help="Parameter file.")
parser.add_argument(
    "--make_swift_param_files", help="Make SWIFT files.", action="store_true"
)
parser.add_argument(
    "--save_pl_data", help="Save fortran PL files.", action="store_true"
)
parser.add_argument(
    "--save_pl_data_hdf5", help="Save HDF5 PL files.", action="store_true"
)
parser.add_argument("--with_mpi", help="Run over MPI.", action="store_true")
args = parser.parse_args()

# Init MPI.
mympi.init_mpi(args.with_mpi)

# Read the parameter file.
if mympi.comm_rank == 0:
    pl_params = ParticleLoadParams(args)
else:
    pl_params = None
if mympi.comm_size > 1:
    pl_params = mympi.comm.bcast(pl_params, root=0)

# Go...

# Load high resolution mask if selected.
if pl_params.is_zoom and pl_params.mask_file is not None:
    pl_params.high_res_region_mask = ZoomRegionMask(pl_params.mask_file)

    # Overwrite the high-res region coordinates using details from the mask file.
    pl_params.coords = pl_params.high_res_region_mask.geo_centre
    print(f"Set coordinates of high res region using mask: {pl_params.coords}")
else:
    pl_params.high_res_region_mask = None

# Generate zoom particle load.
if pl_params.is_zoom:
    if mympi.comm_rank == 0:
        print("\n------ High res grid ------")
    high_res_region = HighResolutionRegion(pl_params)
    # self.high_res_region.plot()

    if mympi.comm_rank == 0:
        print("\n------ Low res skins ------")
    low_res_region = LowResolutionRegion(pl_params, high_res_region)

    n_tot = high_res_region.n_tot + low_res_region.n_tot
    if mympi.comm_size > 1:
        n_tot = mympi.comm.allreduce(n_tot)
    min_ranks = np.true_divide(n_tot, pl_params.max_particles_per_ic_file)

    if mympi.comm_rank == 0:
        print("\n------ Totals (zoom simulation) ------")
        compute_fft_stats(np.max(high_res_region.size_high_res), n_tot, pl_params)
        print(f" --- Total number of particles {n_tot} ({n_tot**(1/3.):.1f} cubed)")
        print(
            f" --- Num ranks needed for less than <max_particles_per_ic_file> = {min_ranks:.2f}"
        )

    # Populate the grid with particles.
    if pl_params.save_pl_data:
        populate_particles(n_tot, high_res_region, low_res_region, pl_params)
else:
    high_res_region = None
    low_res_region = None

    n_tot = pl_params.n_particles
    min_ranks = np.true_divide(n_tot, pl_params.max_particles_per_ic_file)

    if mympi.comm_rank == 0:
        print("\n------ Totals (full volume simulation)------")
        compute_fft_stats(None, n_tot, pl_params)
        print(f" --- Total number of particles {n_tot} ({n_tot**(1/3.):.1f} cubed)")
        print(
            f" --- Num ranks needed for less than <max_particles_per_ic_file> = {min_ranks:.2f}"
        )

# Make the param files.
param_dict = build_param_dict(pl_params, high_res_region)

if pl_params.make_ic_gen_param_files:
    make_param_file_ics(param_dict)
    make_submit_file_ics(param_dict)
    print("Saved ics param and submit file.")

if pl_params.make_swift_param_files:
    # Make swift param file (remember no h's for swift).
    make_param_file_swift(param_dict)
    make_submit_file_swift(param_dict)
    print("Saved swift param and submit file.")
