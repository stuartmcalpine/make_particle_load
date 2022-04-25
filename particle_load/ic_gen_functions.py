import numpy as np


def compute_ic_cores_from_mem(
    nmaxpart, nmaxdisp, ndim_fft, all_ntot, pl_params, optimal=False
):
    ncores_ndisp = np.ceil(
        float((ndim_fft * ndim_fft * 2 * (ndim_fft / 2 + 1))) / nmaxdisp
    )
    ncores_npart = np.ceil(float(all_ntot) / nmaxpart)
    ncores = max(ncores_ndisp, ncores_npart)
    while (ndim_fft % ncores) != 0:
        ncores += 1

    # If we're using one node, try to use as many of the cores as possible
    if ncores < pl_params.ncores_node:
        ncores = pl_params.ncores_node
        while (ndim_fft % ncores) != 0:
            ncores -= 1
    this_str = "[Optimal] " if optimal else ""
    print(
        "%sUsing %i cores for IC gen (min %i for FFT and min %i for particles)"
        % (this_str, ncores, ncores_ndisp, ncores_npart)
    )
    if optimal == False:
        pl_params.n_cores_ic_gen = ncores


def compute_optimal_ic_mem(ndim_fft, all_ntot, pl_params):
    """This will compute the optimal memory to fit IC gen on cosma7."""

    bytes_per_particle = 66.0
    bytes_per_grid_cell = 20.0

    total_memory = (bytes_per_particle * all_ntot) + (
        bytes_per_grid_cell * ndim_fft**3.0
    )

    frac = (bytes_per_particle * all_ntot) / total_memory
    nmaxpart = (frac * pl_params.mem_per_core) / bytes_per_particle

    frac = (bytes_per_grid_cell * ndim_fft**3.0) / total_memory
    nmaxdisp = (frac * pl_params.mem_per_core) / bytes_per_grid_cell

    total_cores = total_memory / pl_params.mem_per_core

    print("[Optimal] nmaxpart= %i nmaxdisp= %i" % (nmaxpart, nmaxdisp))

    compute_ic_cores_from_mem(
        nmaxpart, nmaxdisp, ndim_fft, all_ntot, pl_params, optimal=True
    )


def compute_fft_stats(max_boxsize, all_ntot, pl_params):
    """Work out what size of FFT grid we need for ic_gen."""
    if pl_params.is_zoom:
        if pl_params.is_slab:
            # pl_params.high_res_n_eff = pl_params.n_particles
            # pl_params.high_res_L = pl_params.box_size
            pass
        else:
            # Size of high res region for ic_gen (we use a buffer for saftey).
            high_res_L = pl_params.ic_region_buffer_frac * max_boxsize
            assert high_res_L < pl_params.box_size, "Zoom buffer region too big"

            # Effective number of particles that would fill buffered high res region.
            high_res_n_eff = int(
                pl_params.n_particles * (high_res_L**3.0 / pl_params.box_size**3.0)
            )
        print("HRgrid c=%s L_box=%.2f Mpc/h" % (pl_params.coords, pl_params.box_size))
        print(
            "HRgrid L_grid=%.2f Mpc/h n_eff=%.2f**3 (x2=%.2f**3) FFT buff frac= %.2f"
            % (
                high_res_L,
                high_res_n_eff ** (1 / 3.0),
                2.0 * high_res_n_eff ** (1 / 3.0),
                pl_params.ic_region_buffer_frac,
            )
        )

        # How many multi grid FFT levels, this will update n_eff?
        if pl_params.multigrid_ics:
            if high_res_L > pl_params.box_size / 2.0:
                print("--- Cannot use multigrid ICs, zoom region is > boxsize/2.")
                pl_params.multigrid_ics = 0
            else:
                nlevels = 0
                while pl_params.box_size / (2.0 ** (nlevels + 1)) > high_res_L:
                    nlevels += 1
                actual_high_res_L = pl_params.box_size / (2.0**nlevels)
                assert actual_high_res_L > high_res_L, "Incorrect actual high_res_L"
                actual_high_res_n_eff = int(
                    pl_params.n_particles
                    * (actual_high_res_L**3.0 / pl_params.box_size**3)
                )

                print(
                    "HRgrid num multigrids=%i, lowest=%.2f Mpc/h n_eff: %.2f**3 (x2 %.2f**3)"
                    % (
                        nlevels,
                        actual_high_res_L,
                        actual_high_res_n_eff ** (1 / 3.0),
                        2 * actual_high_res_n_eff ** (1 / 3.0),
                    )
                )
    else:
        high_res_n_eff = pl_params.n_particles
        high_res_L = pl_params.box_size

    pl_params.high_res_L = high_res_L
    pl_params.high_res_n_eff = high_res_n_eff

    # Minimum FFT grid that fits pl_params.fft_times_fac times (defaut=2) the nyquist frequency.
    ndim_fft = pl_params.ndim_fft_start
    N = (high_res_n_eff) ** (1.0 / 3)
    while float(ndim_fft) / float(N) < pl_params.fft_times_fac:
        ndim_fft *= 2
    print("Using ndim_fft = %d" % ndim_fft)

    # Determine number of cores to use based on memory requirements.
    # Number of cores must also be a factor of ndim_fft.
    print("Using nmaxpart= %i nmaxdisp= %i" % (pl_params.nmaxpart, pl_params.nmaxdisp))
    compute_ic_cores_from_mem(
        pl_params.nmaxpart,
        pl_params.nmaxdisp,
        ndim_fft,
        all_ntot,
        pl_params,
        optimal=False,
    )

    # What if we wanted the memory usage to be optimal?
    compute_optimal_ic_mem(ndim_fft, all_ntot, pl_params)
