    def find_nq_slab(self, suggested_nq, slab_width, eps=1.e-4):

        # What is half a slab width? (our starting point)/
        half_slab = slab_width / 2.

        # Dict to save info.
        self.nq_info = {'diff': 1.e20, 'n_tot_lo': 0}

        # What are we starting from?
        if comm_rank == 0:
            print('Computing slab nq: half_slab=%.2f suggested_nq=%i' % (half_slab, suggested_nq))

        # You start with a given suggested nq, then you remove nq_reduce from it at each level.
        # This tries multiple nq_reduce values.
        for nq_reduce in np.arange(1, 10):

            # Loop over an arbitrary number of starting nqs to test them (shouldn't ever need).
            for i in range(200):

                # If this takes me >50% away from the suggested nq, don't bother.
                if np.true_divide(suggested_nq - i, suggested_nq) < 0.5:
                    break

                # Reset all values.
                offset = half_slab  # Starting from the edge of the slab.
                this_nq = suggested_nq - i  # Trying this starting nq.
                starting_nq = suggested_nq - i
                nlev_slab = 0  # Counting the number of levels.
                n_tot_lo = 0  # Counting the total number of low res particles.

                while True:
                    # Cell length at this level.
                    m_int_sep = self.box_size / float(this_nq)

                    # Add this level.
                    offset += m_int_sep
                    nlev_slab += 1

                    # How close are we to the edge.
                    diff = (offset - self.box_size / 2.) / self.box_size

                    # If adding the level has gone outside the box, then stop.
                    # Or if we are really close to the edge.
                    if (offset > self.box_size / 2.) or np.abs(diff) <= 1.e-3:

                        # Try to make it a bit better.

                        # First go back to the previous level.
                        offset -= m_int_sep

                        # Loop over a range of new nq options for the last level/
                        for extra in np.arange(-nq_reduce, nq_reduce, 1):
                            # Try a new nq for the last level.
                            this_nq += extra
                            m_int_sep = self.box_size / float(this_nq)
                            diff = (offset + m_int_sep - self.box_size / 2.) / self.box_size

                            # Have we found a new best level?
                            if np.abs(diff) < np.abs(self.nq_info['diff']):
                                self.nq_info['diff'] = diff
                                self.nq_info['starting_nq'] = starting_nq
                                self.nq_info['finishing_nq'] = this_nq - extra
                                self.nq_info['nq_reduce'] = nq_reduce
                                self.nq_info['extra'] = extra
                                self.nq_info['nlev_slab'] = nlev_slab
                                if (nlev_slab - 1) % comm_size == comm_rank:
                                    n_tot_lo += 2 * this_nq ** 2
                                self.nq_info['n_tot_lo'] = n_tot_lo
                                self.nq_info['dv_slab'] = -1.0 * (
                                        (offset + m_int_sep - self.box_size / 2.) * self.box_size ** 2.
                                ) / self.nq_info['finishing_nq'] ** 2.

                                # Reset for next try.
                                if (nlev_slab - 1) % comm_size == comm_rank:
                                    n_tot_lo -= 2 * this_nq ** 2

                            # Reset for next try.
                            this_nq -= extra
                        break
                    else:
                        if (nlev_slab - 1) % comm_size == comm_rank:
                            n_tot_lo += 2 * this_nq ** 2

                    # Compute nq for next level.
                    this_nq -= nq_reduce

                    # We've gotten too small. 
                    if this_nq < 10:
                        break

        if comm_rank == 0:
            print((
                      '[Rank %i] Best nq: starting_nq=%i, finishing_nq=%i, extra=%i, diff=%.10f, nlev_slab=%i, '
                      'nq_reduce=%i, dv_slab=%.10f n_tot_lo=%i'
                  ) % (
                comm_rank,
                self.nq_info['starting_nq'],
                self.nq_info['finishing_nq'],
                self.nq_info['extra'],
                self.nq_info['diff'],
                self.nq_info['nlev_slab'],
                self.nq_info['nq_reduce'],
                self.nq_info['dv_slab'],
                self.nq_info['n_tot_lo']
            ))
        assert np.abs(self.nq_info['diff']) <= eps, 'Could not find good nq'

        return self.nq_info['n_tot_lo']
