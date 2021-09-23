#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import sys
import ctypes
from numpy cimport ndarray, int32_t, float64_t, float32_t, int64_t
import numpy as np
cimport numpy as np

cdef make_grid(int n_cells_x, int n_cells_y, int n_cells_z, int comm_rank, int comm_size,
        int num_cells):
    cdef int max_n_cells = max(n_cells_x, n_cells_y, n_cells_z)

    cdef ndarray[float64_t, ndim=1, mode="c"] range_lo =\
            np.array(np.arange(0, max_n_cells+1), dtype='f8')

    cdef ndarray[float64_t, ndim=2, mode="c"] offsets = np.zeros((num_cells, 3), dtype='f8')
    cdef ndarray[int32_t, ndim=1, mode="c"] cell_nos = np.empty(num_cells, dtype='i4')

    cdef float lo_x, lo_y, lo_z
    cdef Py_ssize_t i
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t idx_count = 0

    cdef float off = max_n_cells/2.

    for i in range(max_n_cells):
        lo_z = range_lo[i]
        if lo_z >= n_cells_z: continue
        for j in range(max_n_cells):
            lo_y = range_lo[j]
            if lo_y >= n_cells_y: continue
            for k in range(max_n_cells):
                lo_x = range_lo[k]
                if lo_x >= n_cells_x: continue

                if count % comm_size == comm_rank:
                    offsets[idx_count,0] = lo_x - off
                    offsets[idx_count,1] = lo_y - off
                    offsets[idx_count,2] = lo_z - off
                    cell_nos[idx_count] = count
                    idx_count += 1
                count += 1

    assert idx_count == num_cells, 'I did not make the right number of cells'
    offsets[:,0] += (max_n_cells - n_cells_x)/2.
    offsets[:,1] += (max_n_cells - n_cells_y)/2.
    offsets[:,2] += (max_n_cells - n_cells_z)/2.

    return offsets, cell_nos

cdef assign_mask_cells(ndarray[int32_t, ndim=1] cell_types,
        ndarray[float64_t, ndim=2] mask_coords,
        ndarray[float64_t, ndim=2] offsets, double cell_width,
        ndarray[int32_t, ndim=1] cell_nos):
    """ For a given list of coords, see what cells they fill up. """

    cdef int num_cells = len(cell_types)
    cdef long num_mask_particles = len(mask_coords)
    cdef long j
    cdef int i, this_cell, idx, this_type
    cdef double half_cell = cell_width / 2.
    cdef int checknum = num_cells // 4

    for i in range(num_cells):
        for j in range(num_mask_particles):
            if mask_coords[j,0] + half_cell < offsets[i,0]: continue
            if mask_coords[j,0] - half_cell > offsets[i,0] + 1.0: continue
            if mask_coords[j,1] + half_cell < offsets[i,1]: continue
            if mask_coords[j,1] - half_cell > offsets[i,1] + 1.0: continue
            if mask_coords[j,2] + half_cell < offsets[i,2]: continue
            if mask_coords[j,2] - half_cell > offsets[i,2] + 1.0: continue
            cell_types[i] = 0
            break

cdef find_skin_cells(ndarray[int32_t, ndim=1] cell_types,
        ndarray[int32_t, ndim=1] cell_nos, int L, int cell_type):
    """ Find the neighbouring 6 skin cells of each cell in a list. """

    cdef long i, this_cell_no, this_idx
    cdef long count = 0
    cdef skin_cells = np.ones(6*len(cell_nos), dtype='i4') * -1

    cdef mask = np.where(cell_types == cell_type)[0]

    if len(mask) == 0:
        return []
    else:
        for i in range(len(mask)):
            this_idx = mask[i]
            this_cell_no = cell_nos[this_idx]

            if this_cell_no % L != 0:
                skin_cells[count] = this_cell_no - 1
            count += 1
            if this_cell_no % L != L - 1:
                skin_cells[count] = this_cell_no + 1
            count += 1
            if ((this_cell_no / L) % L) != 0:
                skin_cells[count] = this_cell_no - L
            count += 1
            if ((this_cell_no / L) % L) != L - 1:
                skin_cells[count] = this_cell_no + L
            count += 1
            if (this_cell_no / L**2) % L != L - 1:
                skin_cells[count] = this_cell_no + L**2 
            count += 1
            if (this_cell_no / L**2) % L != 0:
                skin_cells[count] = this_cell_no - L**2
            count += 1

        mask = np.where(skin_cells != -1)[0]
        if len(mask) == 0:
            return []
        else:
            return skin_cells[mask]

cdef gen_layered_particles_slab(double slab_width, double boxsize, int nq, int nlev, double dv,
        int comm_rank, int comm_size, int n_tot_lo, int n_tot_hi,
        ndarray[float64_t, ndim=1, mode="c"] coords_x, ndarray[float64_t, ndim=1, mode="c"] coords_y,
        ndarray[float64_t, ndim=1, mode="c"] coords_z, ndarray[float64_t, ndim=1, mode="c"] masses,
        int nq_reduce, int extra):

    cdef double offset, m_int_sep
    cdef int i, j, s, l, idx, this_nq
    cdef double half_slab = slab_width / 2.
    cdef long count = 0

    # Loop over each level.
    offset = half_slab
    count = 0
    for l in range(nlev):

        if l == nlev-1: nq += extra
        
        # Mean interparticle sep at this level.
        m_int_sep = boxsize/float(nq)

        if l % comm_size == comm_rank:
            for s in range(2):              # Both sides of the slab.
                for j in range(nq):
                    for i in range(nq):
                        idx = n_tot_hi + count
                        if l == nlev-1:
                            masses[idx] = m_int_sep**3.+dv
                        else:
                            masses[idx] = m_int_sep**3.
                
                        coords_x[idx] = i*m_int_sep + 0.5*m_int_sep
                        coords_y[idx] = j*m_int_sep + 0.5*m_int_sep
                        if s == 0:
                            coords_z[idx] = boxsize/2. + 0.5*m_int_sep + offset
                        else:
                            coords_z[idx] = boxsize/2. - 0.5*m_int_sep - offset
                        count += 1
        offset += m_int_sep
        nq -= nq_reduce

    assert count == n_tot_lo, 'Rank %i Slab outer particles dont add up %i != %i (nq_reduce=%i extra=%i finishing_nq=%i nlev=%i)'\
            %(comm_rank, count, n_tot_lo, nq_reduce, extra, nq+nq_reduce, nlev)
    if comm_rank == 0: print('Generated %i outer slab layers.'%nlev)
    coords_x[n_tot_hi:] /= boxsize
    coords_x[n_tot_hi:] -= 0.5
    coords_y[n_tot_hi:] /= boxsize
    coords_y[n_tot_hi:] -= 0.5
    coords_z[n_tot_hi:] /= boxsize
    coords_z[n_tot_hi:] -= 0.5
    masses[n_tot_hi:] /= boxsize**3.

cdef _guess_nq(double lbox, int nq, int extra, int comm_rank, int comm_size):
   
    cdef double rat = float(nq)/(nq-2)
    cdef int nlev = int(np.log10(lbox)/np.log10(rat)+0.5)
    cdef int nh = nq - 1
    cdef double total_volume = 0
    cdef int l,i,j,k
    cdef double rlen, rcub

    for l in range(nlev):
        if l % comm_size != comm_rank: continue
        if l == nlev - 1:
            nq += extra
            nh = nq -1
        rlen = rat**l             # Length of this cube.
        rcub = rlen/float(nq-2)   # Length of a cell in the cube.
        
        for k in range(-nh,nh+2,2):
            for j in range(-nh,nh+2,2):
                for i in range(-nh,nh+2,2):
                    ik = max(abs(i),abs(j),abs(k))
                    if ik == nh:
                        total_volume += (rcub)**3.

    return total_volume, nlev

cdef gen_layered_particles(double side, int nq, int comm_rank,
        int comm_size, int n_tot_lo, int n_tot_hi, int extra, double total_volume,
        ndarray[float64_t, ndim=1, mode="c"] coords_x,
        ndarray[float64_t, ndim=1, mode="c"] coords_y,
        ndarray[float64_t, ndim=1, mode="c"] coords_z,
        ndarray[float64_t, ndim=1, mode="c"] masses):
    
    cdef double lbox = 1./side
    cdef double rat
    cdef int nlev
    cdef long count = 0
    cdef int nh = nq - 1 
    cdef int l,i,j,k,ik,idx
    cdef double rlen, rcub

    rat = float(nq)/(nq-2)
    nlev = int(np.log10(lbox)/np.log10(rat)+0.5)
    
    # Difference in volume to make up the mass.
    cdef double dv = (lbox**3. - 1**3. - total_volume) /\
            ((nq-1+extra)**2 * 6 + 2)

    if comm_rank == 0:
        print('Rescaling box to %.4f Mpc/h with nq of %i, extra of %i, over %i levels.'\
                %(lbox, nq, extra, nlev))
    # Loop over each level/skin layer.
    for l in range(nlev):
        if l % comm_size != comm_rank: continue
        if l == nlev - 1:
            nq += extra
            nh = nq -1
        rlen = rat**l           # Length of this cube.
        rcub = rlen/float(nq-2)   # Length of a cell in the cube.
        for k in range(-nh,nh+2,2):
            for j in range(-nh,nh+2,2):
                for i in range(-nh,nh+2,2):
                    ik = max(abs(i),abs(j),abs(k))
                    if ik == nh:
                        idx = n_tot_hi + count
                        if l == nlev-1:
                            masses[idx] = rcub**3.+dv
                        else:
                            masses[idx] = rcub**3.
                        coords_x[idx] = 0.5*rcub*i
                        coords_y[idx] = 0.5*rcub*j
                        coords_z[idx] = 0.5*rcub*k
                        count += 1
    assert count == n_tot_lo, 'Out particles dont add up %i, %i'%(count, n_tot_lo)
    coords_x[n_tot_hi:] /= lbox
    coords_y[n_tot_hi:] /= lbox
    coords_z[n_tot_hi:] /= lbox
    masses[n_tot_hi:] /= lbox**3.

cdef populate_grid(ndarray[float64_t, ndim=2] offsets, ndarray[float64_t, ndim=2] coords_template,
        ndarray[float64_t, ndim=1] coords_x, ndarray[float64_t, ndim=1] coords_y,
        ndarray[float64_t, ndim=1] coords_z, long idx_offset):

    cdef long ncells = len(offsets)
    cdef long ncoords = len(coords_template)
    cdef long count = 0
    cdef long idx
    cdef Py_ssize_t i,j

    for i in range(ncells):
        for j in range(ncoords):
            idx = idx_offset + count
            coords_x[idx] = offsets[i,0] + coords_template[j,0]
            coords_y[idx] = offsets[i,1] + coords_template[j,1]
            coords_z[idx] = offsets[i,2] + coords_template[j,2]
            count += 1

def get_find_skin_cells(cell_types, cell_nos, L, cell_type):
    return find_skin_cells(cell_types, cell_nos, L, cell_type)

def get_guess_nq(lbox, nq, extra, comm_rank, comm_size):
    return _guess_nq(lbox, nq, extra, comm_rank, comm_size)

def get_assign_mask_cells(cell_types, cell_coords, offsets, cell_width, cell_nos):
    return assign_mask_cells(cell_types, cell_coords, offsets, cell_width, cell_nos)

def get_layered_particles_slab(slab_width, boxsize, nq, nlev, dv, comm_rank, comm_size,
        n_tot_lo, n_tot_hi, coords_x, coords_y, coords_z, masses, nq_reduce, extra):
    gen_layered_particles_slab(slab_width, boxsize, nq, nlev, dv, comm_rank, comm_size,
            n_tot_lo, n_tot_hi, coords_x, coords_y, coords_z, masses, nq_reduce, extra)

def get_layered_particles(side, nq, comm_rank, comm_size, n_tot_lo, n_tot_hi,
        extra, total_volume, coords_x, coords_y, coords_z, masses):
    gen_layered_particles(side, nq, comm_rank, comm_size, n_tot_lo, n_tot_hi, extra,
            total_volume, coords_x, coords_y, coords_z, masses)

def get_populated_grid(offsets, coords_template, coords_x, coords_y, coords_z, idx_offset):
    return populate_grid(offsets, coords_template, coords_x, coords_y, coords_z, idx_offset)

def get_grid(n_cells_x, n_cells_y, n_cells_z, comm_rank, comm_size, num_cells):
    return make_grid(n_cells_x, n_cells_y, n_cells_z, comm_rank, comm_size, num_cells)
