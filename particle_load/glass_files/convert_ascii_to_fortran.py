from fortran_functions import read_fortran_glass_file, write_fortran_glass_file, read_ascii_glass_file
import numpy as np
import sys
import os

# Pass file name as a command line argument, eg, "python3 plot_glass.py ascii_glass_4096"
fnames = sys.argv[1:]

for fname in fnames:

    # Load coords from ascii file.
    coords = read_ascii_glass_file(fname)
    print(f"Loaded {len(coords)} coords fro, {fname}")

    # Write these coords to fortan file.
    write_fortran_glass_file(coords)
    print(f"Written to fortran file")

    # Read back in from fortran file.
    newcoords = read_fortran_glass_file(f"fortran_glass_{len(coords)}", 1)
    assert np.array_equal(coords, newcoords), "Arrays not equal"
