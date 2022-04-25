import h5py
import matplotlib.pyplot as plt
import numpy as np

import mympi
from MakeGrid import *
from ZoomRegionMask import ZoomRegionMask


class HighResolutionRegion:
    """Generate cell structure for high-resolution grid."""

    def __init__(self, pl_params):

        # Compute dimensions of the high-res region (in units of glass cells).
        self.set_initial_dimensions(pl_params)

        # Generate the high resolution grid.
        self.init_high_res_region(pl_params)

        # Count up the high resolution particles.
        self.count_high_res_particles(pl_params)

    def plot(self):
        """Have a quick look at how the high resolution region has been constructed."""

        idx = np.abs(self.offsets[:, 2]).argmin()
        mask = np.where(
            (self.offsets[:, 2] == self.offsets[:, 2][idx]) & (self.cell_types == 0)
        )
        plt.scatter(
            self.offsets[:, 0][mask],
            self.offsets[:, 1][mask],
            c=self.cell_types[mask],
            marker="x",
        )
        mask = np.where(
            (self.offsets[:, 2] == self.offsets[:, 2][idx]) & (self.cell_types > 0)
        )
        plt.scatter(
            self.offsets[:, 0][mask], self.offsets[:, 1][mask], c=self.cell_types[mask]
        )
        plt.gca().axis("equal")
        plt.tight_layout(pad=0.1)
        plt.savefig(f"high_res_region_{mympi.comm_rank}.png")
        plt.close()

    def set_initial_dimensions(self, pl_params):
        """Set the initial dimensions of the high-resolution grid."""

        # (Cube root of) how many glass cells would fill the whole simulation volume.
        self.n_cells_whole_volume = int(
            np.rint((pl_params.n_particles / pl_params.glass_num) ** (1 / 3.0))
        )
        assert (
            self.n_cells_whole_volume**3 * pl_params.glass_num
            == pl_params.n_particles
        ), "Error creating high res cell sizes"

        # Size of a glass cell in Mpc/h.
        self.size_glass_cell = np.true_divide(
            pl_params.box_size, self.n_cells_whole_volume
        )

        # (Cube root of) number of glass cells needed to fill the high-res region.
        self.n_cells_high_res = np.tile(
            int(
                np.ceil(
                    pl_params.high_res_region_mask.bounding_length
                    / self.size_glass_cell
                )
            ),
            3,
        )

        # Want a buffer between glass cells and low-res outer shells?
        self.n_cells_high_res += pl_params.glass_buffer_cells * 2
        assert np.all(
            self.n_cells_high_res < self.n_cells_whole_volume
        ), "To many cells"

        # Make sure slabs do the whole box in 2 dimensions.
        # if params.data["is_slab"]:
        #    raise Exception("Test me")
        #    assert n_cells_high[0] == n_cells_high[1], "Only works for slabs in z"
        #    n_cells_high[0] = params.data["n_glass_cells"]
        #    n_cells_high[1] = params.data["n_glass_cells"]

        # Number of cells that cover bounding region.
        # params.data["radius_in_cells"] = np.true_divide(
        #    high_res_mask.params.data["bounding_length"], params.data["glass_cell_length"]
        # )

        # Make sure grid can accomodate the radius factor.
        # if params.data['is_slab == False:
        #    sys.exit('radius factor not tested')
        #    n_cells_high += \
        #        2*int(np.ceil(params.data['radius_factor*params.data['radius_in_cells-params.data['radius_in_cells))

        # What's the width of the slab in cells.
        # if params.data["is_slab"]:
        #    raise Exception("Test me")
        #    # params.data['slab_width_cells'] = \
        #    #    np.minimum(int(np.ceil(bounding_box[2] / cell_length)), n_cells)
        #    # if comm_rank == 0:
        #    #    print('Number of cells for the width of slab = %i' % params.data['slab_width_cells)

        ## Check we have a proper slab.
        # if params.data["is_slab"]:
        #    tmp_mask = np.where(n_cells_high == params.data["n_glass_cells"])[0]
        #    assert len(tmp_mask) == 2, (
        #        "For a slab simulation, 2 dimentions have to fill box "
        #        + "n_cells_high=%s n_cells=%i" % (n_cells_high, n_cells)
        #    )

        # Size of high resolution region in Mpc/h.
        self.size_high_res = [
            self.size_glass_cell * float(self.n_cells_high_res[0]),
            self.size_glass_cell * float(self.n_cells_high_res[1]),
            self.size_glass_cell * float(self.n_cells_high_res[2]),
        ]

        # Volume of high resolution region in (Mpc/h)**3
        self.volume_high_res = np.prod(self.size_high_res)

        # Total number of cells in high-res grid.
        self.n_cells_high_tot = np.prod(self.n_cells_high_res)
        assert (
            self.n_cells_high_tot < 2**32 / 2.0
        ), "Total number of high res cells too big"

        if mympi.comm_rank == 0:
            print(
                f"Dim of high resolution grid {self.size_high_res} Mpc/h",
                f"in {self.n_cells_high_res} ({self.n_cells_high_tot}) cells",
            )

    def compute_offsets(self):
        """Generate the positions of each cell in the high-res region grid.
        Cells are split between cores in MPI."""

        if mympi.comm_rank == 0:
            print(f"Generating high resolution grid {self.n_cells_high_res}...")

        # Number of high-res cells going on this core.
        this_num_cells = self.n_cells_high_tot // mympi.comm_size
        if mympi.comm_rank < (self.n_cells_high_tot) % mympi.comm_size:
            this_num_cells += 1

        # Get offsets and cell_nos for high-res cells on this core.
        offsets, cell_nos = get_grid(
            self.n_cells_high_res[0],
            self.n_cells_high_res[1],
            self.n_cells_high_res[2],
            mympi.comm_rank,
            mympi.comm_size,
            this_num_cells,
        )

        # Make sure we all add up.
        check_num = len(cell_nos)
        assert (
            check_num == this_num_cells
        ), f"Error creating cells {check_num} != {this_num_cells}"

        if mympi.comm_size > 1:
            check_num = mympi.comm.allreduce(check_num)
            assert check_num == self.n_cells_high_tot, "Error creating cells 2."

        return offsets, cell_nos

    def add_high_res_skins(self, cell_nos, cell_types, L):
        """Add degrading skins around glass cells in high res grid."""

        if mympi.comm_rank == 0:
            print("Adding skins around glass cells in high res region...", end="")
        mask = np.where(cell_types == -1)
        num_to_go = len(mask[0])
        if mympi.comm_size > 1:
            num_to_go = mympi.comm.allreduce(num_to_go)
        this_type = 0
        count = 0

        # Loop until we have no more cells to fill.
        while num_to_go > 0:
            # What cells neighbour those at the current level.
            skin_cells = np.unique(
                get_find_skin_cells(cell_types, cell_nos, L, this_type)
            )

            # Share answers over MPI.
            if mympi.comm_size > 1:
                skin_cells_counts = mympi.comm.allgather(len(skin_cells))
                skin_cells_disps = np.cumsum(skin_cells_counts) - skin_cells_counts
                if mympi.comm_rank == 0:
                    skin_cells_all = np.empty(np.sum(skin_cells_counts), dtype="i4")
                else:
                    skin_cells_all = None
                mympi.comm.Gatherv(
                    skin_cells,
                    [skin_cells_all, skin_cells_counts, skin_cells_disps, MPI.INT],
                    root=0,
                )
                if mympi.comm_rank == 0:
                    skin_cells = np.unique(skin_cells_all)
                else:
                    skin_cells = None
                skin_cells = mympi.comm.bcast(skin_cells)

            # Update cell_types for cells at this level.
            idx = np.where(np.in1d(cell_nos, skin_cells))[0]
            idx2 = np.where(cell_types[idx] == -1)
            idx3 = idx[idx2]
            cell_types[idx3] = this_type + 1
            mask = np.where(cell_types == -1)
            if mympi.comm_size > 1:
                num_to_go = mympi.comm.allreduce(len(mask[0]))
            else:
                num_to_go = len(mask[0])
            this_type += 1
            count += 1

        if mympi.comm_rank == 0:
            print("added %i skins." % count)

    def find_nearest_glass_file(self, num, glass_files_dir):
        """In the glass files folder, which one has the closest number of particles."""
        files = os.listdir(glass_files_dir)
        files = np.array(
            [int(x.split("_")[2]) for x in files if "ascii" in x], dtype="i8"
        )
        idx = np.abs(files - num).argmin()
        return files[idx]

    def find_nearest_cube(self, num):
        """Find the nearest number that has a cube root."""
        return int(np.ceil(num ** (1 / 3.0)) ** 3.0)

    def count_high_res_particles(self, pl_params):
        """Count total number of high-resolution particles there will be."""

        self.cell_info = {
            "type": [],
            "num_particles_per_cell": [],
            "num_cells": [],
            "particle_mass": [],
        }
        this_tot_num_glass_particles = 0
        this_tot_num_grid_particles = 0

        # Total number of high-res cells.
        n_tot_cells = len(self.cell_types)
        if mympi.comm_size > 1:
            n_tot_cells = mympi.comm.allreduce(n_tot_cells)

        # Loop over each cell type/particle mass.
        for i in np.unique(self.cell_types):
            num_cells = len(np.where(self.cell_types == i)[0])

            self.cell_info["type"].append(i)
            this_num_cells = len(np.where(self.cell_types == i)[0])
            self.cell_info["num_cells"].append(this_num_cells)

            # Glass particles (type 0).
            if i == 0:
                self.cell_info["num_particles_per_cell"].append(pl_params.glass_num)
                this_tot_num_glass_particles += this_num_cells * pl_params.glass_num
            # Grid particles (type > 0).
            else:
                # Desired number of particles in this level of grid.
                desired_no = np.maximum(
                    pl_params.min_num_per_cell,
                    int(
                        np.ceil(
                            pl_params.glass_num
                            * np.true_divide(pl_params.skin_reduce_factor, i)
                        )
                    ),
                )

                if pl_params.grid_also_glass:
                    # Find glass file with the closest number of particles.
                    num_in_grid_cell = self.find_nearest_glass_file(
                        desired_no, pl_params.glass_files_dir
                    )
                else:
                    # Find nearest cube to this number, for grid.
                    num_in_grid_cell = self.find_nearest_cube(desired_no)

                self.cell_info["num_particles_per_cell"].append(num_in_grid_cell)
                this_tot_num_grid_particles += this_num_cells * num_in_grid_cell

            # Compute the masses of the particles in each cell type.
            self.cell_info["particle_mass"].append(
                (
                    (
                        self.size_glass_cell
                        / self.cell_info["num_particles_per_cell"][-1] ** (1 / 3.0)
                    )
                    / pl_params.box_size
                )
                ** 3.0
            )

        # Make then numpy arrays.
        for att in self.cell_info.keys():
            self.cell_info[att] = np.array(self.cell_info[att])

        # Check we add up.
        n_tot_cells_check = np.sum(self.cell_info["num_cells"])
        if mympi.comm_size > 1:
            n_tot_cells_check = mympi.comm.allreduce(n_tot_cells_check)
        assert n_tot_cells_check == n_tot_cells, "Bad cell count"

        # Total number of glass particles.
        self.tot_num_glass_particles = this_tot_num_glass_particles
        if mympi.comm_size > 1:
            self.tot_num_glass_particles = mympi.comm.allreduce(
                this_tot_num_glass_particles
            )

        # Total number of grid/lower mass glass particles.
        self.tot_num_grid_particles = this_tot_num_grid_particles
        if mympi.comm_size > 1:
            self.tot_num_grid_particles = mympi.comm.allreduce(
                this_tot_num_grid_particles
            )

        # The number of glass particles if they filled the high res grid.
        self.n_tot_glass_part_equiv = pl_params.glass_num * n_tot_cells

        # How many particles in the lowest mass resolution cells in the high res grid.
        num_lowest_res = np.min(self.cell_info["num_particles_per_cell"])
        if mympi.comm_size > 1:
            num_lowest_res = mympi.comm.allreduce(num_lowest_res, op=MPI.MIN)
        self.n_tot_grid_part_equiv = n_tot_cells * num_lowest_res

        # How many levels of resolution are in the high res grid for this rank.
        # self.my_cell_types = np.unique(cell_types)

        # Total number of particles in the high-res grid.
        self.n_tot = self.tot_num_grid_particles + self.tot_num_glass_particles

        # Some global properties of cells
        particle_mass_list = np.unique(self.cell_info["particle_mass"])
        if mympi.comm_size > 1:
            particle_mass_list = np.unique(
                np.concatenate(mympi.comm.allgather(particle_mass_list))
            )

        mask = particle_mass_list != np.min(particle_mass_list)
        self.cell_info["min_grid_mass"] = np.min(particle_mass_list[mask])
        self.cell_info["max_grid_mass"] = np.max(particle_mass_list[mask])
        self.cell_info["glass_mass"] = np.min(particle_mass_list)

        for att in self.cell_info:
            print(f" - cell_info: {att}: {self.cell_info[att]}")

    def init_high_res_region(self, pl_params):
        """Make the high resolution grid."""

        # Generate the grid layout.
        self.offsets, self.cell_nos = self.compute_offsets()

        # Holds the cell types (dictates what will fill them later).
        self.cell_types = (
            np.ones(len(self.offsets), dtype="i4") * -1
        )  # Should all get overwritten.

        L = self.n_cells_high_res[0]  # Number of glass cells on a side.
        max_boxsize = np.max(self.size_high_res)  # Length of high-res region.

        # Using a mask file.
        if pl_params.mask_file is not None:
            # Rescale mask coords into grid coords.
            mask_cell_centers = pl_params.high_res_region_mask.coords / max_boxsize * L
            mask_cell_width = (
                pl_params.high_res_region_mask.grid_cell_width / max_boxsize * L
            )
            assert np.all(np.abs(mask_cell_centers) <= L / 2.0), "Mask coords error"

            if mympi.comm_rank == 0:
                print(
                    "Mask coords x>=%.5f <=%.5f grid cells"
                    % (mask_cell_centers[:, 0].min(), mask_cell_centers[:, 0].max())
                )
                print(
                    "Mask coords y>=%.5f <=%.5f grid cells"
                    % (mask_cell_centers[:, 2].min(), mask_cell_centers[:, 1].max())
                )
                print(
                    "Mask coords z>=%.5f <=%.5f grid cells"
                    % (mask_cell_centers[:, 2].min(), mask_cell_centers[:, 2].max())
                )
                print("Mask cells in grid cells = %.2f" % (mask_cell_width))
                print("Assigning mask cells...")

            # Find out which cells in our grid will be glass given the mask.
            get_assign_mask_cells(
                self.cell_types,
                mask_cell_centers,
                self.offsets,
                mask_cell_width,
                self.cell_nos,
            )

            # Fill the rest with degrading resolution grid or glass cells.
            self.add_high_res_skins(self.cell_nos, self.cell_types, L)

        else:
            raise Exception("Does this work without a mask?")
            # if self.is_slab:
            #    tmp_mask = np.where(np.abs(centers[:,2]) <= (self.slab_width_cells/2.)+1e-4)
            #    cell_types[tmp_mask] = 0
            # else:
            #    if comm_rank == 0: print('Computing distances to each grid cell...')
            #    dists = np.linalg.norm(centers, axis=1)
            #    if comm_rank == 0:
            #        print('Sphere rad %.2f Mpc/h center %s Mpc/h (%.2f Grid cells)'\
            #            %(self.radius, self.coords, self.radius_in_cells))
            #    mask = np.where(dists <= self.radius_factor*self.radius_in_cells)
            #    cell_types[mask] = 0
            #
            #    # Add skins around the glass cells.
            #    self.add_high_res_skins(cell_nos, cell_types, L)
