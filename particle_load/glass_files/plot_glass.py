import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Pass file name as a command line argument, eg, "python3 plot_glass.py ascii_glass_4096"
fnames = sys.argv[1:]

for fname in fnames:
    assert os.path.isfile(fname), f"File {fname} not found"
    
    # Load glass file.
    data = np.loadtxt(fname, dtype={'names': ['x','y','z'], 'formats': ['f4','f4','f4']},
            skiprows=1)
    coords = np.c_[data['x'], data['y'], data['z']]
    
    # Check filename.
    ntot = len(coords)
    assert int(fname.split('_')[-1]) == ntot, "Bad filename"
    
    # Check COM.
    com = np.mean(coords, axis=0)
    assert np.allclose(com, [0.5,0.5,0.5], rtol=1e-05, atol=1e-05), "Bad COM"
    print(f"Centre of mass of glass particles: {com}")
    
    # Plot particles.
    plt.figure(figsize=(3,3))
    plt.title(fname)
    plt.scatter(coords[:,0], coords[:,1], s=1, marker='.')
    plt.xlim(0, 1)
    plt.ylim(0,1)
    plt.tight_layout(pad=0.1)
    plt.savefig(f"./plots/{fname}.png")
    plt.close()
