import h5py
import numpy as np


class ZoomRegionMask:
    """Load the high-resolution region mask file."""

    def __init__(self, mask_file):

        self.mask_file = mask_file

        self.load_mask()

    def load_mask(self):

        print("\n------ Loading mask file ------")
        f = h5py.File(self.mask_file, "r")
        self.coords = np.array(f["Coordinates"][...], dtype="f8")
        self.geo_centre = f["Coordinates"].attrs.get("geo_centre")
        self.bounding_length = f["Coordinates"].attrs.get("bounding_length")
        self.high_res_volume = f["Coordinates"].attrs.get("high_res_volume")
        self.grid_cell_width = f["Coordinates"].attrs.get("grid_cell_width")
        f.close()
        print("Loaded: %s" % self.mask_file)
        print("Mask bounding length = %s Mpc/h" % self.bounding_length)
