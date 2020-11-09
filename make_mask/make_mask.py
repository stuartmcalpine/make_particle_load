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

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "modules"
    )
)

try:
    from peano import peano_hilbert_key_inverses
except ImportError:
    raise Exception("Make sure you have added the `peano.py` module directory to your $PYTHONPATH.")
try:
    from read_swift import read_swift
except ImportError:
    raise Exception("Make sure you have added the `read_swift.py` module directory to your $PYTHONPATH.")
try:
    from read_eagle import EagleSnapshot
except ImportError:
    raise Exception("Make sure you have added the `read_eagle.py` module directory to your $PYTHONPATH.")

comm = MPI.COMM_WORLD
comm_rank = comm.rank
comm_size = comm.size


# try:
#     plt.style.use("../../mnras.mplstyle")
# except:
#     pass

class MakeMask:

    def __init__(self, param_file):

        self.read_param_file(param_file)
        self.make_mask()

    def read_param_file(self, param_file):
        """ Read parameters from YAML file. """
        if comm_rank == 0:
            params = yaml.load(open(param_file))

            # Defaults.
            self.params = {}
            self.params['min_num_per_cell'] = 3
            self.params['mpc_cell_size'] = 3.
            self.params['topology_fill_holes'] = True
            self.params['topology_dilation_niter'] = 0
            self.params['topology_closing_niter'] = 0

            required_params = [
                'fname',
                'snap_file',
                'bits',
                'shape',
                'data_type',
                'divide_ids_by_two',
                'select_from_vr',
                'output_dir'
            ]

            for att in required_params:
                assert att in params.keys(), 'Need to have %s as a param' % att

            # Run checks for automatic and manual group selection
            if params['select_from_vr']:
                assert 'GN' in params.keys(), 'Need to provide a Group-Number for the group'
                assert 'vr_file' in params.keys(), 'Need to provide a sub_file'
                if 'sort_m200crit' in params.keys() and not params['sort_m200crit']:
                    del params['sort_m200crit']
                if 'sort_m500crit' in params.keys() and not params['sort_m500crit']:
                    del params['sort_m500crit']
                assert 'sort_m200crit' in params.keys() or 'sort_m500crit' in params.keys(), \
                    'Need to provide a sort rule for the catalogue by either soring by M200crit ot M500crit.'
                assert 'highres_radius_r200' in params.keys() or 'highres_radius_r500' in params.keys(), \
                    'Need to provide a radius for the high-resolution region in either R200crit ot R500crit units.'

            else:
                assert 'coords' in params.keys(), 'Need to provide coords of region.'
                if params['shape'] == 'cuboid' or params['shape'] == 'slab':
                    assert 'dim' in params.keys(), 'Need to provide dimensions of region.'
                elif params['shape'] == 'sphere':
                    assert 'radius' in params.keys(), 'Need to provide radius of sphere.'

            # Load all parameters into the class
            for att in params.keys():
                self.params[att] = params[att]
        else:
            self.params = None

        self.params = comm.bcast(self.params)

        # Find the group we want to re-simulate (if selected)
        if self.params['select_from_vr']:
            self.params['coords'], self.params['radius'] = self.find_group()
            self.params['shape'] = 'sphere'

        self.params['coords'] = np.array(self.params['coords'], dtype='f8')
        if 'dim' in self.params.keys():
            self.params['dim'] = np.array(self.params['dim'])

    def find_group(self) -> Tuple[List[float], float]:

        is_r200 = 'highres_radius_r200' in self.params.keys()
        is_r500 = 'highres_radius_r500' in self.params.keys()
        is_m200 = 'sort_m200crit' in self.params.keys()
        is_m500 = 'sort_m500crit' in self.params.keys()

        # Warn if conflicts are detected
        if comm_rank == 0:
            if is_r200 and is_r500:
                warn("Conflict: highres_radius_r200 and highres_radius_r500 both specified. Fallback onto R_200crit")
            if is_m200 and is_m500:
                warn("Conflict: sort_m200crit and sort_m500crit both specified. Fallback onto M_200crit")

        # Read in halo properties
        with h5py.File(self.params['vr_file'], 'r') as vr_file:

            structType = vr_file['/Structuretype'][:]
            field_halos = np.where(structType == 10)[0]
            _M200c = vr_file['/Mass_200crit'][field_halos] * 1e10

            # Sort group catalogue by specified rule
            if is_m200:
                sort_key = np.argsort(_M200c)[::-1]  # Reverse for descending order
            elif is_m500 and not is_m200:
                try:
                    _M500c = vr_file['/SO_Mass_500_rhocrit'][field_halos] * 1e10
                    ok_m500 = True
                except KeyError as error:
                    _M500c = _M200c
                    ok_m500 = False
                    if comm_rank == 0:
                        print(error)
                        print("If using sort_m500crit, the sorting rule will use M_200crit instead.")
                        warn("The groups are now sorted by M_200crit.", RuntimeWarning)
                else:
                    sort_key = np.argsort(_M500c)[::-1]  # Reverse for descending order
            else:
                raise ValueError("Mass-sorting keys not correctly configured. Check the parameter file.")

            # Values to strings for printing into log
            M200c = _M200c[sort_key][self.params['GN']]
            if is_m500:
                M500c = _M500c[sort_key][self.params['GN']]
                M500c_str = f"{M500c:.4f}" if ok_m500 else f"? | Fallback M_200crit: {M200c:.4f}"
            else:
                M500c_str = f"None"

            # Construct high resolution radius
            R200c = vr_file['/R_200crit'][field_halos][sort_key][self.params['GN']]
            if is_r200:
                radius = R200c * self.params['highres_radius_r200']
            elif is_r500 and not is_r200:
                try:
                    R500c = vr_file['/SO_R_500_rhocrit'][field_halos][sort_key][self.params['GN']]
                    ok_r500 = True
                except KeyError as error:
                    ok_r500 = False
                    R500c = R200c / 2
                    if comm_rank == 0:
                        print(error)
                        print("If using highres_radius_r500, the selection will use R_200crit instead.")
                        warn("The high-resolution radius is now set to R_200crit * highres_radius_r500 / 2.",
                             RuntimeWarning)
                else:
                    radius = R500c * self.params['highres_radius_r500']
            else:
                raise ValueError("Neither highres_radius_r200 nor highres_radius_r500 were entered.")

            # Values to strings for printing into log
            if is_r500:
                R500c_str = f"{R500c:.4f}" if ok_r500 else f"? | Fallback R_200crit/2: {R500c:.4f}"
            else:
                R500c_str = f"None"

            xPotMin = vr_file['/Xcminpot'][field_halos][sort_key][self.params['GN']]
            yPotMin = vr_file['/Ycminpot'][field_halos][sort_key][self.params['GN']]
            zPotMin = vr_file['/Zcminpot'][field_halos][sort_key][self.params['GN']]

        if comm_rank == 0:
            print(
                "Velociraptor search results:\n",
                f"- Run name: {self.params['fname']}\tGroupNumber: {self.params['GN']}\n",
                f"- Coordinate centre: ", ([xPotMin, yPotMin, zPotMin]), "Mpc\n",
                f"- High-res radius: {radius:.4f} Mpc\n",
                f"- R_200crit: {R200c:.4f} Mpc\n",
                f"- R_500crit: {R500c_str} Mpc\n",
                f"- M_200crit: {M200c:.4f} $M_\odot$\n",
                f"- M_500crit: {M500c_str} $M_\odot$\n"
            )

        return [xPotMin, yPotMin, zPotMin], radius

    def compute_ic_positions(self, ids) -> np.ndarray:
        """ Compute the positions at ICs. """
        print('[Rank %i] Computing initial positions of dark matter particles...' % comm_rank)
        X, Y, Z = peano_hilbert_key_inverses(ids, self.params['bits'])
        ic_coords = np.vstack((X, Y, Z)).T
        assert 0 <= np.all(ic_coords) < 2 ** self.params['bits'], 'Initial coords out of range'
        return np.array(ic_coords, dtype='f8')

    def load_particles(self):
        """ Load particles from chosen snapshot."""

        if self.params['data_type'].lower() == 'gadget':
            snap = EagleSnapshot(self.params['snap_file'])
            self.params['bs'] = float(snap.HEADER['BoxSize'])
            self.params['h_factor'] = 1.0
            self.params['length_unit'] = 'Mph/h'
            self.params['redshift'] = snap.HEADER['Redshift']
        elif self.params['data_type'].lower() == 'swift':
            snap = read_swift(self.params['snap_file'])
            self.params['bs'] = float(snap.HEADER['BoxSize'])
            self.params['h_factor'] = float(snap.COSMOLOGY['h'])
            self.params['length_unit'] = 'Mpc'
            self.params['redshift'] = snap.HEADER['Redshift']

        # A sphere with radius R.
        if self.params['shape'] == 'sphere':
            self.region = [self.params['coords'][0] - self.params['radius'],
                           self.params['coords'][0] + self.params['radius'],
                           self.params['coords'][1] - self.params['radius'],
                           self.params['coords'][1] + self.params['radius'],
                           self.params['coords'][2] - self.params['radius'],
                           self.params['coords'][2] + self.params['radius']]
        # A cuboid with sides x,y,z.
        elif self.params['shape'] == 'cuboid' or self.params['shape'] == 'slab':
            self.region = [self.params['coords'][0] - self.params['dim'][0] / 2.,
                           self.params['coords'][0] + self.params['dim'][0] / 2.,
                           self.params['coords'][1] - self.params['dim'][1] / 2.,
                           self.params['coords'][1] + self.params['dim'][1] / 2.,
                           self.params['coords'][2] - self.params['dim'][2] / 2.,
                           self.params['coords'][2] + self.params['dim'][2] / 2.]
        if comm_rank == 0:
            print('Snapshot is at redshift z=%.2f'%self.params['redshift'])
            print('Loading region...\n', self.region)
        if self.params['data_type'].lower() == 'gadget':
            snap.select_region(*self.region)
            snap.split_selection(comm_rank, comm_size)
        elif self.params['data_type'].lower() == 'swift':
            snap.select_region(1, *self.region)
            snap.split_selection(comm)

        # Load DM particle IDs.
        if comm_rank == 0:
            print('Loading particle data...')
        ids = snap.read_dataset(1, 'ParticleIDs')
        coords = snap.read_dataset(1, 'Coordinates')
        print(f'[Rank {comm_rank}] Loaded {len(ids)} dark matter particles')

        # Wrap coordinates.
        coords = np.mod(coords - self.params['coords'] + 0.5 * self.params['bs'],
                        self.params['bs']) + self.params['coords'] - 0.5 * self.params['bs']

        # Clip to desired shape.
        if self.params['shape'] == 'sphere':
            if comm_rank == 0:
                print('Clipping to sphere around %s, radius %.4f %s' % (
                    self.params['coords'],
                    self.params['radius'],
                    self.params['length_unit']
                ))
            dists = distance.cdist(coords, self.params['coords'].reshape(1, 3), metric='euclidean').reshape(
                len(coords), )
            mask = np.where(dists <= self.params['radius'])

        elif self.params['shape'] == 'cuboid' or self.params['shape'] == 'slab':
            if comm_rank == 0:
                print('Clipping to %s x=%.2f %s, y=%.2f %s, z=%.2f %s around %s %s' \
                      % (self.params['shape'], self.params['dim'][0], self.params['length_unit'],
                         self.params['dim'][1], self.params['length_unit'],
                         self.params['dim'][2], self.params['length_unit'],
                         self.params['coords'], self.params['length_unit']))

            mask = np.where(
                (coords[:, 0] >= (self.params['coords'][0] - self.params['dim'][0] / 2.)) &
                (coords[:, 0] <= (self.params['coords'][0] + self.params['dim'][0] / 2.)) &
                (coords[:, 1] >= (self.params['coords'][1] - self.params['dim'][1] / 2.)) &
                (coords[:, 1] <= (self.params['coords'][1] + self.params['dim'][1] / 2.)) &
                (coords[:, 2] >= (self.params['coords'][2] - self.params['dim'][2] / 2.)) &
                (coords[:, 2] <= (self.params['coords'][2] + self.params['dim'][2] / 2.))
            )

        ids = ids[mask]
        print(f'[Rank {comm_rank}] Clipped to {len(ids)} dark matter particles')

        # Put back into original IDs.
        if self.params['divide_ids_by_two']:
            ids /= 2

        return ids, coords

    def convert_to_inverse_h(self, coords: np.ndarray) -> np.ndarray:
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
        return coords * h

    def make_mask(self):
        # Load particles.
        ids, coords = self.load_particles()
        if self.params['data_type'].lower() == 'swift':
            coords = self.convert_to_inverse_h(coords)

        # Find initial positions from IDs.
        ic_coords = self.compute_ic_positions(ids)

        # Rescale IC coords to 0-->boxsize.
        ic_coords *= np.true_divide(self.params['bs'], 2 ** self.params['bits'] - 1)
        ic_coords = np.mod(ic_coords - self.params['coords'] + 0.5 * self.params['bs'],
                           self.params['bs']) + self.params['coords'] - 0.5 * self.params['bs']

        # Find COM of the lagrangian region.
        count = 0
        last_com_coords = np.array([self.params['bs'] / 2., self.params['bs'] / 2., self.params['bs'] / 2.])

        while True:
            com_coords = self.get_com(ic_coords, self.params['bs'])
            if comm_rank == 0:
                print(f'COM iteration {count} c: {com_coords} Mpc/h')
            ic_coords = np.mod(ic_coords - com_coords + 0.5 * self.params['bs'],
                               self.params['bs']) + com_coords - 0.5 * self.params['bs']
            if np.sum(np.abs(com_coords - last_com_coords)) <= 1e-6:
                break
            last_com_coords = com_coords
            count += 1
            if (count > 10) or (self.params['shape'] == 'slab'):
                break
        if comm_rank == 0:
            print('COM of lagrangian region %s Mpc/h\n\t(compared to coords %s Mpc/h)' \
                  % (com_coords, self.params['coords']))
        ic_coords -= com_coords

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

        # Start with coordinates boundary.
        ic_coord_outline_width = np.max(
            [outline_max_x - outline_min_x,
             outline_max_y - outline_min_y,
             outline_max_z - outline_min_z])

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
            np.abs(np.min(edges[0][m[0]])),
            np.max(edges[0][m[0]]) + bin_width,
            np.abs(np.min(edges[1][m[1]])),
            np.max(edges[1][m[1]]) + bin_width,
            np.abs(np.min(edges[2][m[2]])),
            np.max(edges[2][m[2]]) + bin_width
        ])

        if comm_rank == 0:
            print(
                f"Encompassing dimensions:\n"
                f"\tx = {(lens[0] + lens[1]):.4f} Mpc/h\n"
                f"\ty = {(lens[2] + lens[3]):.4f} Mpc/h\n"
                f"\tz = {(lens[4] + lens[5]):.4f} Mpc/h"
            )

            lens_volume = (lens[0] + lens[1]) * (lens[2] + lens[3]) *\
                (lens[4] + lens[5])
            tot_cells = len(H[0][m[0]])
            tot_cells_volume = tot_cells * bin_width**3.
            print(f'There are {tot_cells:d} total mask cells.')
            print(f'Cells fill {tot_cells_volume/lens_volume:.8f} per cent of region.')

        # Plot the mask and the ameba
        self.plot(H, edges, bin_width, m, ic_coords, lens)

        # Save the mask to hdf5
        if comm_rank == 0:
            self.save(H, edges, bin_width, m, lens, com_coords)

    def plot(self, H, edges, bin_width, m, ic_coords, lens):
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
                rect = patches.Rectangle(
                    (-lens[i * 2], -lens[j * 2]),
                    lens[i * 2 + 1] + lens[i * 2],
                    lens[j * 2 + 1] + lens[j * 2],
                    linewidth=1, edgecolor='r', facecolor='none'
                )

                # Plot particles.
                axarr[count].scatter(plot_coords[:, i], plot_coords[:, j], s=0.5, c='blue',
                        zorder=9, alpha=0.5)
                axarr[count].add_patch(rect)
                if self.params['shape'] == 'slab':
                    if count > 1:
                        axarr[count].set_ylim(-lens[j * 2]+15, -lens[j * 2]-1)
                        axarr[count].set_xlim(-lens[i * 2] - 1, lens[i * 2 + 1] + 1)
                    else:
                        axarr[count].set_ylim(lens[j * 2]-15, lens[j * 2]+1)
                        axarr[count].set_xlim(-lens[i * 2] - 1, lens[i * 2 + 1] + 1)
                else:
                    axarr[count].set_xlim(-lens[i * 2] - 0.05*lens[i * 2],
                            lens[i * 2 + 1] + 0.05*lens[i * 2 + 1])
                    axarr[count].set_ylim(-lens[j * 2] - 0.05*lens[j * 2],
                            lens[j * 2 + 1] + 0.05*lens[j * 2 + 1])

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

    def save(self, H, edges, bin_width, m, lens, com_coords):
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
        ds.attrs.create('xlen_lo', lens[0])
        ds.attrs.create('xlen_hi', lens[1])
        ds.attrs.create('ylen_lo', lens[2])
        ds.attrs.create('ylen_hi', lens[3])
        ds.attrs.create('zlen_lo', lens[4])
        ds.attrs.create('zlen_hi', lens[5])
        ds.attrs.create('coords', self.params['coords'])
        ds.attrs.create('com_coords', com_coords)
        ds.attrs.create('grid_cell_width', bin_width)
        if self.params['shape'] == 'cuboid' or self.params['shape'] == 'slab':
            high_res_volume = self.params['dim'][0] * self.params['dim'][1] * self.params['dim'][2]
        else:
            high_res_volume = 4 / 3. * np.pi * self.params['radius'] ** 3.
        ds.attrs.create('high_res_volume', high_res_volume)
        f.close()
        print(f"Saved {self.params['output_dir']}/{self.params['fname']:s}.hdf5")

    def get_com(self, ic_coords, bs):
        """ Find centre of mass for passed coordinates. """
        if self.params['shape'] == 'slab':
            com_x = bs / 2.
            com_y = bs / 2.
        else:
            com_x = comm.allreduce(np.sum(ic_coords[:, 0])) / comm.allreduce(len(ic_coords[:, 0]))
            com_y = comm.allreduce(np.sum(ic_coords[:, 1])) / comm.allreduce(len(ic_coords[:, 1]))
        com_z = comm.allreduce(np.sum(ic_coords[:, 2])) / comm.allreduce(len(ic_coords[:, 2]))
        return np.array([com_x, com_y, com_z])


if __name__ == '__main__':
    x = MakeMask(sys.argv[1])
