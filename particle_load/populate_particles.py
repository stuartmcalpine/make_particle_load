import numpy as np
import mympi
from MakeGrid import get_populated_grid, get_layered_particles
from save_routines import save_pl


def com(coords_x, coords_y, coords_z, masses):
    """Compute center of mass of a list of coords."""
    com_x = np.sum(coords_x * masses)
    com_y = np.sum(coords_y * masses)
    com_z = np.sum(coords_z * masses)

    if mympi.comm_size > 1:
        return (
            mympi.comm.reduce(com_x),
            mympi.comm.reduce(com_y),
            mympi.comm.reduce(com_z),
        )
    else:
        return com_x, com_y, com_z


def generate_uniform_grid(n_particles):
    """For low res regions just making a uniform grid of particles."""
    if n_particles == 1:
        coords = np.ones((1, 3), dtype="f8") * 0.5
    else:
        L = int(np.rint(n_particles ** (1 / 3.0)))
        coords = np.zeros((n_particles, 3), dtype="f8")
        count = 0
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    coords[count][0] = (i + 0.5) / L
                    coords[count][1] = (j + 0.5) / L
                    coords[count][2] = (k + 0.5) / L
                    count += 1

    assert np.all(coords >= 0.0) and np.all(coords <= 1.0), "Error uniform grid"
    return coords


def load_glass_file(num, pl_params):
    """Load the glass file for high resolution particles."""
    glass = np.loadtxt(
        pl_params.glass_files_dir + "ascii_glass_%i" % num,
        dtype={"names": ["x", "y", "z"], "formats": ["f8", "f8", "f8"]},
        skiprows=1,
    )
    if mympi.comm_rank == 0:
        print("Loaded glass file, %i particles in file." % num)

    return glass


def rescale(x, x_min_old, x_max_old, x_min_new, x_max_new):
    """Rescale an array of numbers to a new min max."""
    return ((x_max_new - x_min_new) / (x_max_old - x_min_old)) * (
        x - x_max_old
    ) + x_max_new


def populate_levels(
    cell_types,
    offsets,
    glass,
    ntot,
    coords_x,
    coords_y,
    coords_z,
    masses,
    pl_params,
    high_res_region,
):
    """This populates the type 0 (glass) and type >0 (grid) cells of the high res grid."""

    # Loop over each cell type and fill up the grid.
    cell_offset = 0
    for i in range(len(high_res_region.cell_info["type"])):
        mask = np.where(cell_types == high_res_region.cell_info["type"][i])
        assert len(mask[0]) > 0, "Dont have types that I should."

        # Glass particles.
        if high_res_region.cell_info["type"][i] == 0:
            get_populated_grid(
                offsets[mask],
                np.c_[
                    glass[pl_params.glass_num]["x"],
                    glass[pl_params.glass_num]["y"],
                    glass[pl_params.glass_num]["z"],
                ],
                coords_x,
                coords_y,
                coords_z,
                cell_offset,
            )

        # Grid particles:
        else:
            if pl_params.grid_also_glass:
                get_populated_grid(
                    offsets[mask],
                    np.c_[
                        glass[high_res_region.cell_info["num_particles_per_cell"][i]][
                            "x"
                        ],
                        glass[high_res_region.cell_info["num_particles_per_cell"][i]][
                            "y"
                        ],
                        glass[high_res_region.cell_info["num_particles_per_cell"][i]][
                            "z"
                        ],
                    ],
                    coords_x,
                    coords_y,
                    coords_z,
                    cell_offset,
                )
            else:
                get_populated_grid(
                    offsets[mask],
                    generate_uniform_grid(
                        high_res_region.cell_info["num_particles_per_cell"][i]
                    ),
                    coords_x,
                    coords_y,
                    coords_z,
                    cell_offset,
                )

        masses[
            cell_offset : cell_offset
            + len(mask[0]) * high_res_region.cell_info["num_particles_per_cell"][i]
        ] = high_res_region.cell_info["particle_mass"][i]

        cell_offset += (
            len(mask[0]) * high_res_region.cell_info["num_particles_per_cell"][i]
        )


def populate_particles(ntot, high_res_region, low_res_region, pl_params):
    # Initiate arrays.
    coords_x = np.empty(ntot, dtype="f8")
    coords_y = np.empty(ntot, dtype="f8")
    coords_z = np.empty(ntot, dtype="f8")
    masses = np.empty(ntot, dtype="f8")

    # Load all the glass files we are going to need.
    glass = {}
    if pl_params.grid_also_glass:
        for this_glass_no in high_res_region.cell_info["num_particles_per_cell"]:
            if this_glass_no not in glass.keys():
                glass[this_glass_no] = load_glass_file(this_glass_no)
    else:
        glass[pl_params.glass_num] = load_glass_file(pl_params.glass_num, pl_params)

    # Populate high resolution grid with particles.
    populate_levels(
        high_res_region.cell_types,
        high_res_region.offsets,
        glass,
        high_res_region.n_tot,
        coords_x,
        coords_y,
        coords_z,
        masses,
        pl_params,
        high_res_region,
    )

    # Rescale masses and coordinates of high res particles and check COM.
    max_cells = high_res_region.n_cells_high_res[0]
    max_boxsize = high_res_region.size_high_res[0]
    coords_x[: high_res_region.n_tot] = (
        rescale(
            coords_x[: high_res_region.n_tot],
            -max_cells / 2.0,
            max_cells / 2.0,
            -max_boxsize / 2,
            max_boxsize / 2.0,
        )
        / pl_params.box_size
    )  # -0.5 > +0.5.
    coords_y[: high_res_region.n_tot] = (
        rescale(
            coords_y[: high_res_region.n_tot],
            -max_cells / 2.0,
            max_cells / 2.0,
            -max_boxsize / 2,
            max_boxsize / 2.0,
        )
        / pl_params.box_size
    )  # -0.5 > +0.5.
    coords_z[: high_res_region.n_tot] = (
        rescale(
            coords_z[: high_res_region.n_tot],
            -max_cells / 2.0,
            max_cells / 2.0,
            -max_boxsize / 2,
            max_boxsize / 2.0,
        )
        / pl_params.box_size
    )  # -0.5 > +0.5.
    assert np.all(
        np.abs(coords_x[: high_res_region.n_tot]) < 0.5
    ), "High res coords error x"
    assert np.all(
        np.abs(coords_y[: high_res_region.n_tot]) < 0.5
    ), "High res coords error y"
    assert np.all(
        np.abs(coords_z[: high_res_region.n_tot]) < 0.5
    ), "High res coords error z"
    tot_hr_mass = np.sum(masses[: high_res_region.n_tot])
    if mympi.comm_size > 1:
        tot_hr_mass = mympi.comm.allreduce(tot_hr_mass)
    assert (
        np.abs(
            tot_hr_mass - (high_res_region.volume_high_res / pl_params.box_size**3.0)
        )
        <= 1e-6
    ), "Error high res masses %.8f" % (
        np.abs(
            tot_hr_mass - (high_res_region.volume_high_res / pl_params.box_size**3.0)
        )
    )
    com_x, com_y, com_z = com(
        coords_x[: high_res_region.n_tot],
        coords_y[: high_res_region.n_tot],
        coords_z[: high_res_region.n_tot],
        masses[: high_res_region.n_tot],
    )
    if mympi.comm_rank == 0:
        print(
            "CoM for high res grid particles [%.2g %.2g %.2g]"
            % (com_x / tot_hr_mass, com_y / tot_hr_mass, com_z / tot_hr_mass)
        )

    # Generate outer particles of low res grid with growing skins.
    if pl_params.is_slab:
        pass
    #        if comm_rank == 0:
    #            print('Putting low res particles around slab of width %.2f Mpc/h' % \
    #                  min_boxsize)
    #        if n_tot_lo > 0:
    #            get_layered_particles_slab(min_boxsize, self.box_size,
    #                                       self.nq_info['starting_nq'], self.nq_info['nlev_slab'],
    #                                       self.nq_info['dv_slab'], comm_rank, comm_size, n_tot_lo, n_tot_hi,
    #                                       coords_x, coords_y, coords_z, masses, self.nq_info['nq_reduce'],
    #                                       self.nq_info['extra'])
    else:
        get_layered_particles(
            low_res_region.side,
            low_res_region.nq_info["nq"],
            mympi.comm_rank,
            mympi.comm_size,
            low_res_region.n_tot,
            high_res_region.n_tot,
            low_res_region.nq_info["extra"],
            low_res_region.nq_info["total_volume"],
            coords_x,
            coords_y,
            coords_z,
            masses,
        )

    max_lr = np.max(masses[high_res_region.n_tot :])
    min_lr = np.min(masses[high_res_region.n_tot :])
    num_lr = len(masses[high_res_region.n_tot :])

    if mympi.comm_size > 1:
        max_lr = comm.allreduce(max_lr, op=MPI.MAX)
        min_lr = comm.allreduce(min_lr, op=MPI.MIN)
        num_lr = comm.allreduce(num_lr)

    #    if mympi.comm_rank == 0:
    #        print(
    #            'Total %i particles in low res region (%.2f cubed) MinM = %.2f (%.2g Msol) MaxM = %.2f (%.2g Msol)' % \
    #            (num_lr, num_lr ** (1 / 3.), np.log10(min_lr), min_lr * self.total_box_mass,
    #             np.log10(max_lr), max_lr * self.total_box_mass))
    #    self.lr_mass_cut = min_lr
    # Checks.
    assert np.all(
        np.abs(coords_x[high_res_region.n_tot :]) <= 0.5
    ), "Low res coords x error"
    assert np.all(
        np.abs(coords_y[high_res_region.n_tot :]) <= 0.5
    ), "Low res coords y error"
    assert np.all(
        np.abs(coords_z[high_res_region.n_tot :]) <= 0.5
    ), "Low res coords z error"

    final_tot_mass = np.sum(masses)

    if mympi.comm_size > 1:
        final_tot_mass = comm.allreduce(final_tot_size)
    tmp_tol = np.abs(1 - final_tot_mass)
    assert tmp_tol <= 1e-5, "Final mass error %.8f != 0.0" % tmp_tol
    if tmp_tol > 1e-6 and mympi.comm_rank == 0:
        print("***Warming*** total final mass tol is %.8f" % tmp_tol)
    com_x, com_y, com_z = com(coords_x, coords_y, coords_z, masses)
    if mympi.comm_rank == 0:
        print(
            "CoM for all particles [%.2g %.2g %.2g]"
            % (com_x / final_tot_mass, com_y / final_tot_mass, com_z / final_tot_mass)
        )

    # final_tot_num = len(masses)
    # final_tot_num = comm.allreduce(len(masses))
    # assert final_tot_num == comm.allreduce(ntot), 'Final array length error'
    # if mympi.comm_rank == 0:
    #    print('Created %i (%.2f cubed) total particles.' \
    #          % (final_tot_num, final_tot_num ** (1 / 3.)))

    # Wrap coords to chosen center.
    wrap_coords = rescale(pl_params.coords, 0, pl_params.box_size, 0, 1.0)
    coords_x = np.mod(coords_x + wrap_coords[0] + 1.0, 1.0)
    coords_y = np.mod(coords_y + wrap_coords[1] + 1.0, 1.0)
    coords_z = np.mod(coords_z + wrap_coords[2] + 1.0, 1.0)

    # Check coords and masses.
    assert np.all(coords_x > 0) and np.all(coords_x < 1.0), "Coords x wrap error"
    assert np.all(coords_y > 0) and np.all(coords_y < 1.0), "Coords y wrap error"
    assert np.all(coords_z > 0) and np.all(coords_z < 1.0), "Coords z wrap error"
    assert np.all(masses > 0.0) and np.all(masses < 1.0), "Mass number error"

    save_pl(coords_x, coords_y, coords_z, masses, pl_params)
