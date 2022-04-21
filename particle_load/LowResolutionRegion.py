import numpy as np
import mympi
from MakeGrid import get_guess_nq


class LowResolutionRegion:
    """Layered skin structure for low res region."""

    def __init__(self, pl_params, high_res_region):

        if pl_params.is_slab:
            self.compute_skins_slab()
        else:
            self.compute_skins(pl_params, high_res_region)

        # Report findings.
        self.print_nq_info()

    def compute_skins_slab(self):
        # Starting nq is equiv of double the mass of the most massive grid particles.
        # suggested_nq = \
        #    int(num_lowest_res ** (1 / 3.) * max_cells * self.nq_mass_reduce_factor)
        # n_tot_lo = self.find_nq_slab(suggested_nq, slab_width)
        pass

    def compute_skins(self, pl_params, high_res_region):

        # A high nq could make boundary particles less massive than high-res, avoid this.
        pl_params.max_nq = np.minimum(
            int(np.floor(high_res_region.n_tot_grid_part_equiv ** (1 / 3.0))),
            pl_params._max_nq,
        )

        # Starting nq is equiv of double the mass of the most massive grid particles.
        suggested_nq = np.clip(
            int(
                high_res_region.n_tot_grid_part_equiv ** (1 / 3.0)
                * pl_params.nq_mass_reduce_factor
            ),
            pl_params.min_nq,
            pl_params.max_nq,
        )

        if mympi.comm_rank == 0:
            print(
                f"Starting: nq={suggested_nq}",
                f"(min/max bounds={pl_params.min_nq}/{pl_params.max_nq})",
            )

        # Compute nq
        self.side = np.true_divide(
            high_res_region.n_cells_high_res[0],
            high_res_region.n_cells_whole_volume,
        )
        self.find_nq(self.side, suggested_nq)

        # Record total number of low res particles.
        self.n_tot = self.nq_info["n_tot_lo"]

    def print_nq_info(self):
        for att in self.nq_info.keys():
            print(f" - nq_info: {att}: {self.nq_info[att]}")

    def find_nq(self, side, suggested_nq, eps=0.01):
        """Estimate what the best value of nq should be."""

        self.nq_info = {"diff": 1.0e20}
        lbox = 1.0 / side

        # Loop over a range of potential nq's.
        for nq in np.arange(suggested_nq - 5, suggested_nq + 5, 1):
            if nq < 10:
                continue
            found_good = 0

            # Loop over a range of extras.
            for extra in range(-10, 10, 1):
                if nq + extra < 10:
                    continue

                # For this nq and extra, what volume would the particles fill.
                total_volume, nlev = get_guess_nq(
                    lbox, nq, extra, mympi.comm_rank, mympi.comm_size
                )
                if mympi.comm_size > 1:
                    total_volume = mympi.comm.allreduce(total_volume)

                # How does this volume compare to the volume we need to fill?
                diff = np.abs(1 - (total_volume / (lbox**3.0 - 1.0**3)))

                if diff < self.nq_info["diff"]:
                    self.nq_info["diff"] = diff
                    self.nq_info["nq"] = nq
                    self.nq_info["extra"] = extra
                    self.nq_info["nlev"] = nlev
                    self.nq_info["total_volume"] = total_volume

        assert self.nq_info["diff"] <= eps, "Did not find good nq. (diff = %.6f)" % (
            self.nq_info["diff"]
        )

        # Compute low res particle number for this core.
        n_tot_lo = 0
        for l in range(self.nq_info["nlev"]):
            if l % mympi.comm_size != mympi.comm_rank:
                continue
            if l == self.nq_info["nlev"] - 1:
                n_tot_lo += (
                    self.nq_info["nq"] - 1 + self.nq_info["extra"]
                ) ** 2 * 6 + 2
            else:
                n_tot_lo += (self.nq_info["nq"] - 1) ** 2 * 6 + 2
        self.nq_info["n_tot_lo"] = n_tot_lo
