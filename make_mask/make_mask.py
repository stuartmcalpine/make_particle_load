"""
Script to generate a mask for a single object.
"""

import sys
import os
import yaml
import h5py
import numpy as np
from typing import List, Tuple
from warnings import warn
from scipy.spatial import distance
from scipy import ndimage
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpi4py import MPI

from pdb import set_trace

# ---------------------------------------
# Load utilities from `modules` directory
# ---------------------------------------

# Append modules directory to PYTHONPATH
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "modules"
    )
)

# Load utilities, with safety checks to make sure that they are accessible
try:
    from peano import peano_hilbert_key_inverses
except ImportError:
    raise Exception(
        "Make sure you have added the `peano.py` module directory to your "
        "$PYTHONPATH.")
try:
    from read_swift import read_swift
except ImportError:
    raise Exception(
        "Make sure you have added the `read_swift.py` module directory to "
        "your $PYTHONPATH.")
try:
    from read_eagle import EagleSnapshot
except ImportError:
    raise Exception(
        "Make sure you have added the `read_eagle.py` module directory to "
        "your $PYTHONPATH.")


# Set up MPI support
comm = MPI.COMM_WORLD
comm_rank = comm.rank
comm_size = comm.size


class MakeMask:
    """
    Class to construct and save a mask.

    Upon instantiation, the parameter file is read, the mask is created
    and (by default) immediately saved to an HDF5 file.

    The class also contains a `plot` method for generating an overview
    plot of the generated mask.
    
    Parameters
    ----------
    param_file : string
        The name of the YAML parameter file defining the mask.
    save : bool
        Switch to directly save the generated mask as an HDF5 file. This is
        True by default.

    Returns
    -------
    None

    """
    def __init__(self, param_file, save=True):

        # Parse the parameter file, check for consistency, and determine
        # the centre and radius of high-res sphere around a VR halo if desired.
        self.read_param_file(param_file)

        # Create the actual mask...
        self.make_mask()

    def read_param_file(self, param_file):
        """
        Read parameters from a specified YAML file.

        See template file `param_files/template.yml` for a full listing and
        description of parameters. The values are stored in the internal
        `self.params` dict.

        If the parameter file specifies the target centre in terms of a
        particular Velociraptor halo, the centre and radius of the high-res
        region are determined internally.

        In the MPI version, the file is only read on one rank and the dict
        then broadcast to all other ranks.
        
        """
        if comm_rank == 0:
            params = yaml.safe_load(open(param_file))

            # Set default values for optional parameters
            self.params = {}
            self.params['min_num_per_cell'] = 3
            self.params['mpc_cell_size'] = 3.
            self.params['topology_fill_holes'] = True
            self.params['topology_dilation_niter'] = 0
            self.params['topology_closing_niter'] = 0
            
            # Define a list of parameters that must be provided. An error
            # is raised if they are not found in the YAML file.
            required_params = [
                'fname',
                'snap_file',
                'bits',
                'data_type',
                'divide_ids_by_two',
                'select_from_vr',
                'output_dir'
            ]
            for att in required_params:
                assert att in params.keys(), (
                    f"Need to provide a value for {att} in the parameter "
                    "file '{param_file}'!")
            
            # Run checks for automatic group selection
            if params['select_from_vr']:
                params['shape'] = 'sphere'
                requirements = [
                    ('group_number', 'a group number to select'),
                    ('vr_file',
                     'a Velociraptor catalogue to select groups from'),
                    ('sort_type', 'the method for halo sorting')
                    ]
                for req in requirements:
                    assert req[0] in params.keys(), (
                     f'Need to provide {req[1]}!')

                # Make sure that we have a positive high-res region size
                if 'highres_radius_r200' not in params:
                    params['highres_radius_r200'] = 0
                if 'highres_radius_r500' not in params:
                    params['highres_radius_r500'] = 0
                if max(params['highres_radius_r200'],
                       params['highres_radius_r500']) <= 0:
                    raise ValueError(
                        "At least one of 'highres_radius_r200' and "
                        "highres_radius_r500' must be positive!")

                # Set defaults for optional parameters
                self.params['r_highres_min'] = 0
                self.params['r_highres_buffer'] = 0
                
            else:
                # Consistency checks for manual target region selection
                assert 'coords' in params.keys(), (
                    "Need to provide coordinates of target region centre!")
                assert 'shape' in params.keys(), (
                    "Need to specify the shape of the target region!")
                if params['shape'] in ['cuboid', 'slab']:
                    assert 'dim' in params.keys(), (
                        f"Need to provide dimensions of '{params[shape]}' "
                        f"high-resolution region.")
                elif params['shape'] == 'sphere':
                    assert 'radius' in params.keys(), (
                        "Need to provide the radius of target "
                        "high-resolution sphere!")

            # Load all parameters into the class
            for att in params.keys():
                self.params[att] = params[att]

            # If desired, find the halo to center the high-resolution on
            if self.params['select_from_vr']:
                self.params['coords'], self.params['radius'] = (
                    self.find_highres_sphere())

            # Convert coordinates and cuboid/slab dimensions to ndarray
            self.params['coords'] = np.array(self.params['coords'], dtype='f8')
            if 'dim' in self.params:
                self.params['dim'] = np.array(self.params['dim'])
                
        else:
            # If this is not the root rank, don't read the file.
            self.params = None

        # Broadcast the read and processed dict to all ranks.
        self.params = comm.bcast(self.params)

    def find_highres_sphere(self) -> Tuple[np.ndarray, float]:
        """
        Determine the centre and radius of high-res sphere from Velociraptor.

        The selection is made based on the location of the halo in the
        catalogue, optionally after sorting them by M200c or M500c. This
        is determined by the value of `self.params['sort_type']`.

        This function is only executed by the root rank.

        Returns
        -------
        centre : ndarray(float)
            3-element array holding the centre of the high-res region.
        radius : float
            The target radius of the high-res region, including any requested
            padding.
            
        """
        # Make sure that we are on the root rank if over MPI
        if comm_rank != 0:
            raise ValueError(
                f"find_highres_sphere() called on MPI rank {comm_rank}!")

        # Look up the target halo index in Velociraptor catalogue
        vr_index = self.find_halo_index()

        with h5py.File(self.params['vr_file'], 'r') as vr_file:

            # First, determine the radius of the high-res region
            r_r200 = 0
            r_r500 = 0
            r200 = vr_file['R_200crit'][vr_index]
            if self.params['highres_radius_r200'] > 0:
                r_r200 = r200 * self.params['highres_radius_r200']
            try:
                r500 = vr_file['SO_R_500_rhocrit'][vr_index]
                r_r500 = r500 * self.params['highres_radius_r500']
            except KeyError:
                r500 = None
                if self.params['highres_radius_r500'] > 0:
                    warn("Could not load R500c, ignoring request for "
                         f"minimum high-res radius of "
                         f"{self.params['highres_radius_r500']} r_500.",
                         RuntimeWarning)

            r_highres = max(r_r200, r_r500, self.params['r_highres_min'])
            if r_highres <= 0:
                raise ValueError(
                    f"Invalid radius of high-res region ({r_highres})")

            # If enabled, add a fixed "padding" radius to the high-res sphere
            if self.params["r_highres_padding"] > 0:
                r_highres += self.params['r_highres_padding']

            # Load halo centre
            names = ['X', 'Y', 'Z']
            centre = np.zeros(3)
            for icoord, prefix in enumerate(names):
                centre[icoord] = vr_file[f'{prefix}cminpot'][vr_index]

        r500_str = '' if r500 is None else f'{r500:.4f}'
        m200_str = (
            '' if getattr(self, 'M200crit', None) is None else
            f'{self.M200crit:.4f}')
        m500_str = (
            '' if getattr(self, 'M500crit', None) is None else
            f'{self.M500crit:.4f}')
        print(
            "Velociraptor search results:\n"
            f"- Run name: {self.params['fname']}\t"
            f"GroupNumber: {self.params['group_number']}\n"
            f"- Centre: {centre[0]:.3f} / {centre[1]:.3f} / {centre[2]:.3f} "
            f"- High-res radius: {r_highres:.4f}\n"
            f"- R_200crit: {r200:.4f}\n"
            f"- R_500crit: {r500_str}\n"
            f"- M_200crit: {m200_str}\n"
            f"- M_500crit: {m500_str}\n"
            )

        return centre, r_highres


    def find_halo_index(self) -> int:
        """
        Find the index of the desired target halo.

        This function looks up the desired (field) halo if the selection
        is specified in terms of a position in the mass-ordered list.
        It should only ever be run on the root node, an error is raised if
        this is not the case.

        If the parameter file instructs to sort by M500c, but this is not
        recorded in the Velociraptor catalogue, an error is raised.
        
        Parameters
        ----------
        None

        Returns
        -------
        halo_index : int
            The catalogue index of the target halo.
        
        """
        if comm_rank != 0:
            raise ValueError("find_halo_index() called on rank {comm_rank}!")

        # If the parameter file already specified the VR index, we are done
        if self.sort_rule.lower() == "none":
            return self.params['group_number']

        # ... otherwise, need to load the desired mass type of all (central)
        # VR haloes, sort them, and find the entry we want
        with h5py.File(self.params['vr_file'], 'r') as vr_file:
            structType = vr_file['/Structuretype'][:]
            field_halos = np.where(structType == 10)[0]

            sort_rule = self.params['sort_type']
            if sort_rule == 'M200crit':
                m_halo = vr_file['/Mass_200crit'][field_halos]
            elif sort_rule == 'M500crit':
                # If M500 cannot be loaded, an error will be raised
                m_halo = vr_file['/SO_Mass_500_rhocrit'][field_halos]
            else:
                raise ValueError("Unknown sorting rule '{sort_rule}'!")
                
        # Sort groups by specified mass, in descending order
        sort_key = np.argsort(-m_halo)
        halo_index = sort_key[self.params['group_number']]

        # Store mass of target halo used for sorting, for later use
        setattr(self, sort_rule, m_halo[halo_index])
        return halo_index

    def make_mask(self):
        """
        Main driver function to create a mask from a given snapshot file.

        This assumes that the centre and extent of the high-res region
        have already been determined, either from the parameter file or
        from the Velociraptor halo catalogue.

        """
        # Find cuboidal frame enclosing the target high-resolution region
        self.region = self.find_enclosing_frame()
        
        # Load IDs of particles within target high-res region from snapshot
        ids = self.load_particles()

        # Find initial positions from IDs
        ic_coords = self.compute_ic_positions(ids)

        # Rescale IC coords to 0-->boxsize.
        # Move this to compute_ic_positions...
        ic_coords *= np.true_divide(self.params['bs'], 2 ** self.params['bits'] - 1)
        ic_coords = np.mod(
            ic_coords - self.params['coords'] + 0.5 * self.params['bs'],
            self.params['bs']
            ) + self.params['coords'] - 0.5 * self.params['bs']

        # Compute outline of histogram region.
        if len(ic_coords) == 0:
            outline_min_x = outline_min_y = outline_min_z = 1.e20
            outline_max_x = outline_max_y = outline_max_z = -1.e20
        else:
            outline_min_x = np.min(ic_coords[:,0])
            outline_min_y = np.min(ic_coords[:,1])
            outline_min_z = np.min(ic_coords[:,2])

            outline_max_x = np.max(ic_coords[:,0])
            outline_max_y = np.max(ic_coords[:,1])
            outline_max_z = np.max(ic_coords[:,2])

        outline_min_x = comm.allreduce(outline_min_x, op=MPI.MIN)
        outline_max_x = comm.allreduce(outline_max_x, op=MPI.MAX)
        outline_min_y = comm.allreduce(outline_min_y, op=MPI.MIN)
        outline_max_y = comm.allreduce(outline_max_y, op=MPI.MAX)
        outline_min_z = comm.allreduce(outline_min_z, op=MPI.MIN)
        outline_max_z = comm.allreduce(outline_max_z, op=MPI.MAX)

        dx = outline_max_x - outline_min_x
        dy = outline_max_y - outline_min_y
        dz = outline_max_z - outline_min_z

        # Put coordinates relative to geometric center.
        geo_centre = np.array([outline_min_x + dx/2.,
                               outline_min_y + dy/2.,
                               outline_min_z + dz/2.])
        ic_coords -= geo_centre

        # Start with coordinates boundary.
        ic_coord_outline_width = np.max([dx,dy,dz])

        # Region can be buffered, x2 to be safe.
        ic_coord_outline_width *= 2

        # Can't be bigger than the box size.
        outline_width = min(ic_coord_outline_width, self.params['bs'])

        # Create mask.
        num_bins = \
            int(np.ceil(outline_width / (self.params['mpc_cell_size'])))
        bins = np.linspace(-outline_width / 2., outline_width / 2., num_bins)
        bin_width = bins[1] - bins[0]
        H, edges = np.histogramdd(ic_coords, bins=(bins, bins, bins))
        H = comm.allreduce(H)

        # Initialize binary mask
        bin_mask = np.zeros_like(H, dtype=np.bool)
        m = np.where(H >= self.params['min_num_per_cell'])
        bin_mask[m] = True

        # Fill holes and extrude the mask
        if comm_rank == 0:
            print("[Topological extrusion (1/3)] Scanning x-y plane...")
        for layer_id in range(bin_mask.shape[0]):
            if self.params['topology_fill_holes']:
                bin_mask[layer_id, :, :] = ndimage.binary_fill_holes(
                    bin_mask[layer_id, :, :]
                ).astype(np.bool)
            if self.params['topology_dilation_niter'] > 0:
                bin_mask[layer_id, :, :] = ndimage.binary_dilation(
                    bin_mask[layer_id, :, :],
                    iterations=self.params['topology_dilation_niter']
                ).astype(np.bool)
            if self.params['topology_closing_niter'] > 0:
                bin_mask[layer_id, :, :] = ndimage.binary_closing(
                    bin_mask[layer_id, :, :],
                    iterations=self.params['topology_closing_niter']
                ).astype(np.bool)

        if comm_rank == 0:
            print("[Topological extrusion (2/3)] Scanning y-z plane...")
        for layer_id in range(bin_mask.shape[1]):
            if self.params['topology_fill_holes']:
                bin_mask[:, layer_id, :] = ndimage.binary_fill_holes(
                    bin_mask[:, layer_id, :],
                ).astype(np.bool)
            if self.params['topology_dilation_niter'] > 0:
                bin_mask[:, layer_id, :] = ndimage.binary_dilation(
                    bin_mask[:, layer_id, :],
                    iterations=self.params['topology_dilation_niter']
                ).astype(np.bool)
            if self.params['topology_closing_niter'] > 0:
                bin_mask[:, layer_id, :] = ndimage.binary_closing(
                    bin_mask[:, layer_id, :],
                    iterations=self.params['topology_closing_niter']
                ).astype(np.bool)

        if comm_rank == 0:
            print("[Topological extrusion (3/3)] Scanning x-z plane...")
        for layer_id in range(bin_mask.shape[2]):
            if self.params['topology_fill_holes']:
                bin_mask[:, :, layer_id] = ndimage.binary_fill_holes(
                    bin_mask[:, :, layer_id],
                ).astype(np.bool)
            if self.params['topology_dilation_niter'] > 0:
                bin_mask[:, :, layer_id] = ndimage.binary_dilation(
                    bin_mask[:, :, layer_id],
                    iterations=self.params['topology_dilation_niter']
                ).astype(np.bool)
            if self.params['topology_closing_niter'] > 0:
                bin_mask[:, :, layer_id] = ndimage.binary_closing(
                    bin_mask[:, :, layer_id],
                    iterations=self.params['topology_closing_niter']
                ).astype(np.bool)

        # Computing bounding region
        m = np.where(bin_mask == True)
        lens = np.array([
            np.min(edges[0][m[0]]),
            np.max(edges[0][m[0]]) + bin_width,
            np.min(edges[1][m[1]]),
            np.max(edges[1][m[1]]) + bin_width,
            np.min(edges[2][m[2]]),
            np.max(edges[2][m[2]]) + bin_width
        ])
        dx = np.max((np.abs(lens[0])*2, np.abs(lens[1])*2))
        dy = np.max((np.abs(lens[2])*2, np.abs(lens[3])*2))
        dz = np.max((np.abs(lens[4])*2, np.abs(lens[5])*2))

        bounding_length = np.max([dx,dy,dz])

        if comm_rank == 0:
            print(
                f"Encompassing dimensions:\n"
                f"\tx = {dx:.4f} Mpc/h\n"
                f"\ty = {dy:.4f} Mpc/h\n"
                f"\tz = {dz:.4f} Mpc/h\n"
                f"Bounding length: {bounding_length:.4f} Mpc/h"
            )

            lens_volume = dx*dy*dz
            tot_cells = len(H[0][m[0]])
            tot_cells_volume = tot_cells * bin_width**3.
            print(f'There are {tot_cells:d} total mask cells.')
            print(f'Cells fill {tot_cells_volume/lens_volume:.8f} per cent of region.')

        # Make sure everything is in the bounding box.
        assert np.all(np.abs(ic_coords)+bin_width/2. <= bounding_length/2.),\
                'Not everything in bounding box'

        # Plot the mask and the amoeba
        self.plot(H, edges, bin_width, m, ic_coords, bounding_length)

        # Save the mask to hdf5
        if comm_rank == 0:
            self.save(H, edges, bin_width, m, bounding_length, geo_centre)

    def find_enclosing_frame(self):
        """
        Compute the bounding box enclosing the target high-res region.

        This is only used to pre-select a region for particle reading from
        the snapshot, to make things more efficient.

        Returns
        -------
        frame : np.ndarray(float)
            A 2x3 element array containing the lower and upper coordinates
            of the bounding region in the x, y, and z coordinates.

        """
        frame = np.zeros((2, 3))
        # If the target region is a sphere, find the enclosing cube
        if self.params['shape'] == 'sphere':
            frame[0, :] = self.params['coords'] - self.params['radius']
            frame[1, :] = self.params['coords'] + self.params['radius']

        # If the target region is a cuboid, simply transform from centre and
        # side length to lower and upper bounds along each coordinate
        elif self.params['shape'] in ['cuboid', 'slab']:
            frame[0, :] = self.params['coords'] - self.params['dim'] / 2.
            frame[1, :] = self.params['coords'] + self.params['dim'] / 2.
    
        print(f"Boundary frame extent:\n"
              f"{frame[0, 0]:.2f} / {frame[0, 1]:.2f} / {frame[0, 2]:.2f} "
              f"-- {frame[1, 0]:.2f} / {frame[1, 1]:.2f} / {frame[1, 2]:.2f}")
            
        return frame
            
    def load_particles(self):
        """
        Load relevant data from base snapshot.

        In addition to particle IDs and coordinates, relevant metadata are
        also loaded and stored in the `self.params` dict.

        Returns
        -------
        ids : ndarray(int)
            The Peano-Hilbert keys of the particles.
        """

        # To make life simpler, extractsome frequently used parameters
        cen = self.params['coords']
        shape = self.params['shape']

        # First step: set up particle reader and load metadata.
        # This is different for SWIFT and GADGET simulations, but subsequent
        # loading 
        if self.params['data_type'].lower() == 'gadget':
            snap = EagleSnapshot(self.params['snap_file'])
            self.params['bs'] = float(snap.HEADER['BoxSize'])
            self.params['h_factor'] = 1.0
            self.params['length_unit'] = 'Mph/h'
            self.params['redshift'] = snap.HEADER['Redshift']
            snap.select_region(*self.region.T.flatten())
            snap.split_selection(comm_rank, comm_size)

        elif self.params['data_type'].lower() == 'swift':
            snap = read_swift(self.params['snap_file'])
            self.params['bs'] = float(snap.HEADER['BoxSize'])
            self.params['h_factor'] = float(snap.COSMOLOGY['h'])
            self.params['length_unit'] = 'Mpc'
            self.params['redshift'] = snap.HEADER['Redshift'][0]   
            snap.select_region(1, *self.region.T.flatten())
            snap.split_selection(comm)

        if comm_rank == 0:
            print(f"Snapshot is at redshift z={self.params['redshift']:.2f}.")

        # Load DM particle IDs and coordinates (uniform across GADGET/SWIFT)
        if comm_rank == 0:
            print('Loading particle data...')
        coords = snap.read_dataset(1, 'Coordinates')

        # Shift coordinates relative to target centre, and wrap them to within
        # the periodic box (done by first shifting them up by half a box,
        # taking the modulus with the box size in each dimension, and then
        # shifting it back down by half a box)
        cen = self.params['coords']
        coords = (
            (coords - cen + 0.5 * self.params['bs']) % self.params['bs']
            - 0.5 * self.params['bs']
            )
        # Select particles within target region
        l_unit = self.params['length_unit']
        if shape == 'sphere':
            if comm_rank == 0:
                print(f"Clipping to sphere around {cen}, with radius "
                      f"{self.params['radius']:.4f} {l_unit}"
                      f"{self.params['length_unit']}"
                      )
            dists = np.linalg.norm(coords, axis=1)
            mask = np.where(dists <= self.params['radius'])[0]

        elif self.params['shape'] in ['cuboid', 'slab']:
            if comm_rank == 0:
                print(f"Clipping to {shape} with "
                      f"dx={self.params['dim'][0]:.2f} {l_unit}, "
                      f"dy={self.params['dim'][1]:.2f} {l_unit}, "
                      f"dz={self.params['dim'][2]:.2f} {l_unit}\n"
                      f"around {cen} {l_unit}.")

            # To find particles within target cuboid, normalize each coordinate
            # offset by the maximum allowed extent in the corresponding
            # dimension, and find those where the result is between -1 and 1
            # for all three dimensions
            mask = np.where(
                np.max(np.abs(coords / (self.params['dim'] / 2)), axis=1)
                <= 1)[0]
            
        # Secondly, we need the IDs of particles lying in the mask region
        ids = snap.read_dataset(1, 'ParticleIDs')[mask]

        # If IDs are Peano-Hilbert indices multiplied by two (as in e.g.
        # simulations with baryons), need to undo this multiplication here
        if self.params['divide_ids_by_two']:
            ids = ids // 2

        print(f'[Rank {comm_rank}] Loaded {len(ids)} dark matter particles')

        # If the snapshot is from a user-friendly SWIFT simulation, all
        # lengths are in 'h-free' coordinates. Unfortunately, the ICs still
        # assume 'h^-1' units, so for consistency we now have to multiply
        # h factor back in...
        if self.params['data_type'].lower() == 'swift':
            self.convert_lengths_to_inverse_h()
             
        return ids
    
    def convert_lengths_to_inverse_h(self):
        """
        Convert length parameters into 'h^-1' units for consistency with ICs.
        """
        h = self.params['h_factor']
        keys = self.params.keys()
        if 'radius' in keys:
            self.params['radius'] *= h
        if 'dim' in keys:
            self.params['dim'] *= h
        if 'coords' in keys:
            self.params['coords'] *= h
        if 'bs' in keys:
            self.params['bs'] *= h
        if 'mpc_cell_size' in keys:
            self.params['mpc_cell_size'] *= h

    def compute_ic_positions(self, ids) -> np.ndarray:
        """
        Compute the particle positions in the ICs.

        This exploits the fact that the particle IDs are set to Peano-Hilbert
        keys that encode the positions in the ICs.

        Parameters
        ----------
        ids : ndarray(int)
            The Particle IDs (more specifically: Peano-Hilbert keys) for which
            to calculate IC coordinates.

        Returns
        -------
        coords : ndarray(float)
            The coordinates of particles in the ICs. Unclear in what shape...?

        """
        print(f"[Rank {comm_rank}] Computing initial positions of dark matter "
               "particles...")

        # Converting the PH keys back to positions is done using an
        # external utility function.
        X, Y, Z = peano_hilbert_key_inverses(ids, self.params['bits'])
        ic_coords = np.vstack((X, Y, Z)).T

        # Make sure that we get consistent values for the coordinates
        assert 0 <= np.all(ic_coords) < 2 ** self.params['bits'], (
            'Initial coords out of range!')
        return np.array(ic_coords, dtype='f8')

            
    def plot(self, H, edges, bin_width, m, ic_coords, bounding_length):
        """ Plot the region outline. """
        axes_label = ['x', 'y', 'z']

        # What's the width of the slab?
        if self.params['shape'] == 'slab':
            slab_width = min(
                    self.region[1] - self.region[0],
                    self.region[3] - self.region[2],
                    self.region[5] - self.region[4]) * self.params['h_factor']

        # Subsample.
        idx = np.random.permutation(len(ic_coords))
        if len(ic_coords) > 1e5:
            idx = np.random.permutation(len(ic_coords))
            plot_coords = ic_coords[idx][:100000]
        else:
            plot_coords = ic_coords[idx]

        plot_coords = comm.gather(plot_coords)

        # Core 0 plots.
        if comm_rank == 0:
            plot_coords = np.vstack(plot_coords)
            if self.params['shape'] == 'slab':
                fig, axarr = plt.subplots(4, 1, figsize=(20, 12))
            else:
                fig, axarr = plt.subplots(1, 3, figsize=(10, 4))

            count = 0
            for i, j in zip([0, 0, 1, 0, 1], [1, 2, 2, 2, 2]):
                if self.params['shape'] == 'slab' and j != 2: continue
                if self.params['shape'] != 'slab' and count > 2: break
                if self.params['shape'] != 'slab': axarr[count].set_aspect('equal')

                # This outlines the bounding region.
                rect = patches.Rectangle([-bounding_length/2., -bounding_length/2.],
                    bounding_length, bounding_length,
                    linewidth=1, edgecolor='r', facecolor='none'
                )

                # Plot particles.
                axarr[count].scatter(plot_coords[:, i], plot_coords[:, j], s=0.5, c='blue',
                        zorder=9, alpha=0.5)
                axarr[count].add_patch(rect)
                if self.params['shape'] == 'slab':
                    pass
                    #if count > 1:
                    #    axarr[count].set_ylim(-lens[j * 2]+15, -lens[j * 2]-1)
                    #    axarr[count].set_xlim(-lens[i * 2] - 1, lens[i * 2 + 1] + 1)
                    #else:
                    #    axarr[count].set_ylim(lens[j * 2]-15, lens[j * 2]+1)
                    #    axarr[count].set_xlim(-lens[i * 2] - 1, lens[i * 2 + 1] + 1)
                else:
                    axarr[count].set_xlim(-(bounding_length/2.)*1.05, (bounding_length/2.)*1.05)
                    axarr[count].set_ylim(-(bounding_length/2.)*1.05, (bounding_length/2.)*1.05)

                # Plot cell bin centers.
                axarr[count].scatter(
                    edges[i][m[i]] + bin_width / 2.,
                    edges[j][m[j]] + bin_width / 2.,
                    marker='x', color='red', s=3, alpha=0.4
                )

                # Plot cell outlines if there isn't too many of them.
                if len(m[i]) < 10000:
                    for e_x, e_y in zip(edges[i][m[i]], edges[j][m[j]]):
                        rect = patches.Rectangle(
                            (e_x, e_y),
                            bin_width,
                            bin_width,
                            linewidth=0.5,
                            edgecolor='r',
                            facecolor='none'
                        )
                        axarr[count].add_patch(rect)

                axarr[count].set_xlabel(f"{axes_label[i]} [Mpc h$^{{-1}}$]")
                axarr[count].set_ylabel(f"{axes_label[j]} [Mpc h$^{{-1}}$]")
                count += 1

            # Plot target sphere.
            if self.params['shape'] == 'sphere':
                for i in range(3):
                    rect = patches.Circle((0,0), radius=self.params['radius'],
                        linewidth=1, edgecolor='k', facecolor='none', ls='--',
                        zorder=10)
                    axarr[i].add_patch(rect)

            plt.tight_layout(pad=0.1)
            plt.savefig(f"{self.params['output_dir']}/{self.params['fname']:s}.png")
            plt.close()

    def save(self, H, edges, bin_width, m, bounding_length, geo_centre):
        # Save (everything needs to be saved in h inverse units, for the IC GEN).
        f = h5py.File(f"{self.params['output_dir']}/{self.params['fname']:s}.hdf5", 'w')
        coords = np.c_[edges[0][m[0]] + bin_width / 2.,
                       edges[1][m[1]] + bin_width / 2.,
                       edges[2][m[2]] + bin_width / 2.]

        # Push parameter file data as file attributes
        g = f.create_group('Params')
        for param_attr in self.params:
            g.attrs.create(param_attr, self.params[param_attr])

        ds = f.create_dataset('Coordinates', data=np.array(coords, dtype='f8'))
        ds.attrs.create('bounding_length', bounding_length)
        ds.attrs.create('geo_centre', geo_centre)
        ds.attrs.create('grid_cell_width', bin_width)
        if self.params['shape'] == 'cuboid' or self.params['shape'] == 'slab':
            high_res_volume = self.params['dim'][0] * self.params['dim'][1] * self.params['dim'][2]
        else:
            high_res_volume = 4 / 3. * np.pi * self.params['radius'] ** 3.
        ds.attrs.create('high_res_volume', high_res_volume)
        f.close()
        print(f"Saved {self.params['output_dir']}/{self.params['fname']:s}.hdf5")

# Allow using the file as stand-alone script
if __name__ == '__main__':
    x = MakeMask(sys.argv[1])
