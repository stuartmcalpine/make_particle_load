import numpy as np
import h5py
import os


class read_swift:

    def __init__(self, fname, verbose=False):
        self.verbose = verbose

        # These get allocated when select_region is called.
        self.lefts = None
        self.rights = None
        self.num_to_load = 0

        # This will get switched on is split selection is called.
        self.mpi = False

        # Snapshot file to load.
        self.fname = fname
        assert os.path.isfile(self.fname), 'File %s does not exist.' % fname

        # Get information from the header.
        self.read_header()

    def split_selection(self, comm):
        """ Splits selected particles between cores. """
        assert self.lefts is not None and self.rights is not None, \
            'Need to call select region before split_selection'

        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        if comm_size > 1:
            # How many particles will each core load?
            num_per_core = self.num_to_load // comm_size
            offset = 0
            if comm_rank == comm_size - 1: offset += self.num_to_load % comm_size
            my_num_to_load = num_per_core + offset

            # What particles will each core load?
            my_lefts = []
            my_rights = []
            count = 0  # How many total particles have been loaded.
            my_count = np.zeros(comm_size, dtype='i8')  # How many particles have I loaded?
            for i, (l, r, chunk_no) in enumerate(zip(self.lefts, self.rights, self.rights - self.lefts)):
                mask = np.where(my_count < num_per_core)

                # How many cores will this chunk be spread over?
                num_cores_this_chunk = ((chunk_no + my_count[np.min(mask)]) // num_per_core) + 1
                chunk_bucket = chunk_no
                for j in range(num_cores_this_chunk):
                    if np.min(mask) + j < comm_size:
                        if my_count[np.min(mask) + j] + chunk_bucket > num_per_core:
                            if comm_rank == np.min(mask) + j:
                                my_lefts.append(l + chunk_no - chunk_bucket)
                                my_rights.append(l + chunk_no - chunk_bucket + \
                                                 num_per_core - my_count[np.min(mask) + j])
                            chunk_bucket -= (num_per_core - my_count[np.min(mask) + j])
                            my_count[np.min(mask) + j] = num_per_core
                        else:
                            if comm_rank == np.min(mask) + j:
                                my_lefts.append(l + chunk_no - chunk_bucket)
                                my_rights.append(r)
                            my_count[np.min(mask) + j] += chunk_bucket
                    else:
                        if comm_rank == comm_size - 1:
                            if my_count[-1] < my_num_to_load:
                                my_rights[-1] += chunk_bucket

            # Store new limits.
            self.lefts = my_lefts
            self.rights = my_rights
            self.num_to_load = my_num_to_load

            # Make sure we got them all.
            assert comm.allreduce(np.sum(np.array(self.rights) - np.array(self.lefts))) \
                   == comm.allreduce(self.num_to_load), 'Did not divide up the particles correctly'

            self.mpi = True
            self.comm = comm
            if self.verbose:
                comm.barrier()
                print('Rank %i will load %i particles l=%s r=%s' \
                      % (comm_rank, self.num_to_load, self.lefts, self.rights))

    def select_region(self, part_type, x_min, x_max, y_min, y_max, z_min, z_max, eps=1e-4, just_load_all=False):
        """ Select what cells contrain the particles in a cube around passed coordinates. """
        if self.verbose:
            print('Selecting on PartType %i' % part_type)
            print('Selection region x=%.4f->%.4f y=%.4f->%.4f z=%.4f->%.4f' \
                  % (x_min, x_max, y_min, y_max, z_min, z_max))

        f = h5py.File(self.fname, "r")
        if 'Cells' in f:
            centres = f["/Cells/Centres"][...]
            size = f["/Cells/Meta-data"].attrs["size"]

            # Coordinates to load around.
            coords = np.array([x_min + (x_max - x_min) / 2.,
                               y_min + (y_max - y_min) / 2.,
                               z_min + (z_max - z_min) / 2.])

            # Wrap to given coordinates.
            boxsize = self.HEADER['BoxSize']
            centres = np.mod(centres - coords + 0.5 * boxsize, boxsize) + coords - 0.5 * boxsize

            # Find what cells fall within boundary.
            dx_over_2 = (x_max - x_min) / 2. + eps
            dy_over_2 = (y_max - y_min) / 2. + eps
            dz_over_2 = (z_max - z_min) / 2. + eps
            half_size = size / 2.  # Half a cell size.

            mask = np.logical_and(
                centres[:, 0] + half_size[0] >= coords[0] - dx_over_2, np.logical_and(
                    centres[:, 0] - half_size[0] <= coords[0] + dx_over_2, np.logical_and(
                        centres[:, 1] + half_size[1] >= coords[1] - dy_over_2, np.logical_and(
                            centres[:, 1] - half_size[1] <= coords[1] + dy_over_2, np.logical_and(
                                centres[:, 2] + half_size[2] >= coords[2] - dz_over_2,
                                centres[:, 2] - half_size[2] <= coords[2] + dz_over_2
                            )
                        )
                    )
                )
            )
        else:
            just_load_all = True
            f.close()

        if just_load_all:
            # Just load everything, don't use the Cells group.
            self.lefts = [0]
            self.rights = [self.HEADER['NumPart_Total'][part_type]]
            self.num_to_load = self.HEADER['NumPart_Total'][part_type]

        elif len(np.where(mask)[0]) > 0:
            if "/Cells/OffsetsInFile/" in f:
                offsets = f["/Cells/OffsetsInFile/PartType%i" % part_type][mask]
            else:
                offsets = f["/Cells/Offsets/PartType%i" % part_type][mask]
            counts = f["/Cells/Counts/PartType%i" % part_type][mask]
            f.close()

            # Only interested in cells with at least 1 particle.
            mask = np.where(counts > 0)
            offsets = offsets[mask]
            counts = counts[mask]

            if self.verbose:
                print('%i cells selected, a cell size is %s.' % (len(offsets), size))

            # Case of no cells.
            if len(offsets) == 0:
                raise ('No particles found in selected region.')
            # Case of one cell.
            elif len(offsets) == 1:
                self.lefts = [offsets[0]]
                self.rights = [offsets[0] + counts[0]]
            # Case of multiple cells.
            else:
                self.lefts = []
                self.rights = []

                buff = 0
                for i in range(len(offsets) - 1):
                    if offsets[i] + counts[i] == offsets[i + 1]:
                        buff += counts[i]

                        if i == len(offsets) - 2:
                            self.lefts.append(offsets[i + 1] - buff)
                            self.rights.append(offsets[i + 1] + counts[i + 1])
                    else:
                        self.lefts.append(offsets[i] - buff)
                        self.rights.append(offsets[i] + counts[i])
                        buff = 0

                        if i == len(offsets) - 2:
                            self.lefts.append(offsets[i + 1] - buff)
                            self.rights.append(offsets[i + 1] + counts[i + 1])

            self.lefts = np.array(self.lefts)
            self.rights = np.array(self.rights)
            self.num_to_load = np.sum(counts)

            # Make sure we found all the particles we intended to.
            assert np.sum(self.rights - self.lefts) == self.num_to_load, \
                'Error loading region, count err'
        else:
            f.close()
            raise ('No particles found in selected region.')

    def read_dataset(self, parttype, att, physical_cgs=False):
        # Compute how many particles this core will load.
        assert self.HEADER['NumPart_Total'][parttype] > 0, 'No particles found.'
        assert self.lefts is not None and self.rights is not None, \
            'Need to select region before loading particles.'

        # How many particles are we loading (on this core)?
        if self.mpi:
            tot = self.comm.allreduce(self.num_to_load)
        else:
            tot = self.num_to_load
        if self.verbose:
            print('[%s] Loading %i of %i particles.' \
                  % (att, tot, self.HEADER['NumPart_Total'][parttype]))

        # Find shape of array we're interested in.
        if self.mpi:
            f = h5py.File(self.fname, 'r', driver='mpio', comm=self.comm)
        else:
            f = h5py.File(self.fname, 'r')
        shape = f['PartType%i/%s' % (parttype, att)].shape
        dtype = f['PartType%i/%s' % (parttype, att)].dtype

        if len(shape) > 1:
            return_array = np.empty((self.num_to_load, shape[1]), dtype=dtype)
        else:
            return_array = np.empty(self.num_to_load, dtype=dtype)

        # Populate return array.
        count = 0
        for l, r in zip(self.lefts, self.rights):
            return_array[count:count + r - l] = f['PartType%i/%s' % (parttype, att)][l:r]
            count += r - l

        # Convert?
        if physical_cgs:
            fac = f['PartType%i/%s' % (parttype, att)].attrs.get(
                "Conversion factor to physical CGS (including cosmological corrections)")
            return_array *= fac

        f.close()

        return return_array

    def read_header(self):
        """ Get information from the header and cosmology. """
        f = h5py.File(self.fname, "r")
        self.HEADER = {}
        for att in f['Header'].attrs.keys():
            self.HEADER[att] = f['Header'].attrs.get(att)
        if 'BoxSize' in self.HEADER.keys():
            try:
                self.HEADER['BoxSize'] = float(self.HEADER['BoxSize'][0])
            except:
                pass

        if 'Cosmology' in f:
            self.COSMOLOGY = {}
            for att in f['Cosmology'].attrs.keys():
                self.COSMOLOGY[att] = f['Cosmology'].attrs.get(att)
        f.close()
