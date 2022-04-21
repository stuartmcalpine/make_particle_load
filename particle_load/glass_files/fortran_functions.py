from scipy.io import FortranFile
import numpy as np
import os 

def read_ascii_glass_file(fname):
    assert os.path.isfile(fname), f"File {fname} not found"

    # Load glass file.
    data = np.loadtxt(fname, dtype={'names': ['x','y','z'],
        'formats': [np.float64, np.float64, np.float64]}, skiprows=1)
    coords = np.c_[data['x'], data['y'], data['z']]
    
    # Check filename.
    ntot = len(coords)
    assert int(fname.split('_')[-1]) == ntot, "Bad filename"
    
    # Check COM.
    com = np.mean(coords, axis=0)
    assert np.allclose(com, [0.5,0.5,0.5], rtol=1e-05, atol=1e-05), "Bad COM"
    print(f"Centre of mass of glass particles: {com}")

    return coords

def read_fortran_glass_file(fname, n_files):

    count = 0
    ntot = None

    for i in range(n_files):
        # Check file part exists.
        tmp_fname = fname + f".{i}"
        assert os.path.isfile(tmp_fname), f"File {tmp_fname} not found"

        # Load this file part.
        f = FortranFile(tmp_fname, "r")
    
        header = f.read_ints(np.int32)
        x = f.read_reals(np.float64)
        y = f.read_reals(np.float64)
        z = f.read_reals(np.float64)
        assert len(x) == len(y) == len(z), "Bad read length"
        assert n_files == header[4], "Bad n_files against header"
        assert len(x) == header[0], "Coords length error with file"

        f.close()
    
        if ntot is None:
            ntot = header[1]
            coords = np.zeros((ntot,3), dtype=np.float64)
        else:
            assert ntot == header[1], "Bad ntot"

        # Append coords from this file.
        coords[count:count+len(x),0] = x
        coords[count:count+len(y),1] = y
        coords[count:count+len(z),2] = z

        count += header[0]

    # Check we add up.
    assert count == ntot, "Don't add up"

    # Check COM.
    com = np.mean(coords, axis=0)
    assert np.allclose(com, [0.5,0.5,0.5], rtol=1e-05, atol=1e-05), "Bad COM"
    print(f"Centre of mass of glass particles: {com}")

    return coords

def write_fortran_glass_file(coords):
    assert coords.dtype == np.float64, f"Wrong coords type, {coords.dtype}"
    
    ntot = len(coords)
    fname = f"fortran_glass_{ntot}.0"

    f = FortranFile(fname, "w")

    header = np.array([ntot, ntot, 0, 0, 1, 10001, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    f.write_record(header)
    f.write_record(coords[:,0])
    f.write_record(coords[:,1])
    f.write_record(coords[:,2])
    f.close()


if __name__ == "__main__":
    fname = "/cosma7/data/dp004/arj/projects/Eagle_200/glass/Eagle_glass_file_47"
    read_fortran_glass_file(fname,32)

    coords = np.random.rand(100,3)
    write_fortran_glass_file(coords)

    coordsnew = read_fortran_glass_file('fortran_glass_100',1)
    assert ip.array_equal(caords,coordsnew), "Bad array match"
