import numpy as np
import matplotlib.pyplot as plt
import sys
from fortran_functions import read_ascii_glass_file, read_fortran_glass_file

# Pass file name as a command line argument, eg, "python3 plot_glass.py ascii_glass_4096"
fnames = sys.argv[1:]

for fname in fnames:

    print(f"Plotting {fname}...")
    if "ascii_glass_" in fname:
        coords = read_ascii_glass_file(fname)
    elif "fortran_glass_" in fname:
        coords = read_fortran_glass_file(fname, 1)
    else:
        raise ValueError("Bad fname")
    
    # Plot particles.
    plt.figure(figsize=(5,5))
    plt.title(fname)
    plt.scatter(coords[:,0], coords[:,1], s=1, marker='.')
    plt.xlim(0, 1)
    plt.ylim(0,1)
    plt.tight_layout(pad=0.1)
    plt.savefig(f"./plots/{fname}.png")
    plt.close()
