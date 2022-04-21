from parallel_functions import repartition
import os
import numpy as np
import mympi
from scipy.io import FortranFile
import h5py


def save_particle_load_as_binary(
    fname, coords_x, coords_y, coords_z, masses, n_tot, nfile, nfile_tot
):
    f = FortranFile(fname, mode="w")
    # 4+8+4+4+4 = 24
    f.write_record(
        np.int32(coords_x.shape[0]),
        np.int64(n_tot),
        np.int32(nfile),
        np.int32(nfile_tot),
        np.int32(0),
        # Now we pad the header with 6 zeros to make the header length
        # 48 bytes in total
        np.int32(0),
        np.int32(0),
        np.int32(0),
        np.int32(0),
        np.int32(0),
        np.int32(0),
    )
    f.write_record(coords_x.astype(np.float64))
    f.write_record(coords_y.astype(np.float64))
    f.write_record(coords_z.astype(np.float64))
    f.write_record(masses.astype("float32"))
    f.close()


def save_pl(coords_x, coords_y, coords_z, masses, pl_params):
    ntot = len(masses)
    if mympi.comm_size > 1:
        ntot = comm.allreduce(ntot)

    # Load balance.
    if mympi.comm_size > 1:
        ndesired = np.zeros(mympi.comm_size, dtype=int)
        ndesired[:] = ntot / mympi.comm_size
        ndesired[-1] += ntot - sum(ndesired)
        if mympi.comm_rank == 0:
            tmp_num_per_file = ndesired[0] ** (1 / 3.0)
            print(
                "Load balancing %i particles on %i ranks (%.2f**3 per file)..."
                % (ntot, mympi.comm_size, tmp_num_per_file)
            )
            if tmp_num_per_file > pl_params.max_particles_per_ic_file ** (1 / 3.0):
                print(
                    "***WARNING*** more than %s per file***"
                    % (pl_params.max_particles_per_ic_file)
                )

        masses = repartition(
            masses, ndesired, mympi.comm, mympi.comm_rank, mympi.comm_size
        )
        coords_x = repartition(
            coords_x, ndesired, mympi.comm, mympi.comm_rank, mympi.comm_size
        )
        coords_y = repartition(
            coords_y, ndesired, mympi.comm, mympi.comm_rank, mympi.comm_size
        )
        coords_z = repartition(
            coords_z, ndesired, mympi.comm, mympi.comm_rank, mympi.comm_size
        )
        mympi.comm.barrier()
        if mympi.comm_rank == 0:
            print("Done load balancing.")

    assert (
        len(masses) == len(coords_x) == len(coords_y) == len(coords_z)
    ), "Array length error"

    """ Save particle load to HDF5 file. """
    save_dir = "%s/ic_gen_submit_files/%s/particle_load/" % (
        pl_params.ic_dir,
        pl_params.f_name,
    )
    save_dir_hdf = save_dir + "hdf5/"
    save_dir_bin = save_dir + "fbinary/"
    if mympi.comm_rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir_hdf) and pl_params.save_pl_data_hdf5:
            os.makedirs(save_dir_hdf)
        if not os.path.exists(save_dir_bin):
            os.makedirs(save_dir_bin)
    if mympi.comm_size > 1:
        mympi.comm.barrier()

    # Make sure not to save more than max_save at a time.
    max_save = 50
    lo_ranks = np.arange(0, mympi.comm_size + max_save, max_save)[:-1]
    hi_ranks = np.arange(0, mympi.comm_size + max_save, max_save)[1:]
    for lo, hi in zip(lo_ranks, hi_ranks):
        if mympi.comm_rank >= lo and mympi.comm_rank < hi:

            if pl_params.save_pl_data_hdf5:
                if mympi.comm_rank == 0:
                    print("Saving HDF5 files...")
                f = h5py.File(save_dir_hdf + "PL.%d.hdf5" % mympi.comm_rank, "w")
                g = f.create_group("PartType1")
                g.create_dataset("Coordinates", (len(masses), 3), dtype="f8")
                g["Coordinates"][:, 0] = coords_x
                g["Coordinates"][:, 1] = coords_y
                g["Coordinates"][:, 2] = coords_z
                g.create_dataset("Masses", data=masses)
                g.create_dataset("ParticleIDs", data=np.arange(0, len(masses)))
                g = f.create_group("Header")
                g.attrs.create("nlist", len(masses))
                g.attrs.create("itot", ntot)
                g.attrs.create("nj", mympi.comm_rank)
                g.attrs.create("nfile", mympi.comm_size)
                g.attrs.create("coords", pl_params.coords / pl_params.box_size)
                g.attrs.create("radius", pl_params.radius / pl_params.box_size)
                g.attrs.create(
                    "cell_length", pl_params.cell_length / pl_params.box_size
                )
                g.attrs.create("Redshift", 1000)
                g.attrs.create("Time", 0)
                g.attrs.create("NumPart_ThisFile", [0, len(masses), 0, 0, 0])
                g.attrs.create("NumPart_Total", [0, ntot, 0, 0, 0])
                g.attrs.create("NumPart_TotalHighWord", [0, 0, 0, 0, 0])
                g.attrs.create("NumFilesPerSnapshot", comm_size)
                g.attrs.create("ThisFile", comm_rank)
                f.close()

            # Save to fortran binary.
            fname = "%s/PL.%d" % (save_dir_bin, mympi.comm_rank)
            save_particle_load_as_binary(
                fname,
                coords_x,
                coords_y,
                coords_z,
                masses,
                ntot,
                mympi.comm_rank,
                mympi.comm_size,
            )

            print(
                "[%i] Saved %i/%i particles..." % (mympi.comm_rank, len(masses), ntot)
            )
