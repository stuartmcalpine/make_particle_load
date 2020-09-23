Set-up this repository
--------
You can clone this repository locally by simply running 
```bash
git clone https://github.com/stuartmcalpine/make_particle_load.git
```
and using
```bash
git pull
```
to update to the latest changes. The masking and particle-load
scripts rely on several dependencies, collected in the `modules` sub-directory. In order to run
the scripts successfully, you will need to activate these modules and make the `modules` directory
visible to the `python3` executable. The process of setting the correct `PYTHONPATH` is 
automatically done by running 
```bash
source setup.sh
```
in the top-level directory tree. 

Dependencies
--------
The `modules` directory contains the following dependencies:

| Package name           | Platform   | Utility                                                            | Compiled binaries                                 |
|:---------------------- |------------|--------------------------------------------------------------------|---------------------------------------------------|
| `peano.py`             | Python | Translates particle IDs into coordinates from the initial conditions.  | None                                              |
| `read_swift.py`        | Python | Reads snapshots generated using the SWIFT code.                        | None                                              |
| `read_eagle.py`        | Python/C | Reads snapshots generated using the GADGET code in EAGLE format.     | `ReadEagle-*.egg-info` `_read_eagle.cpython*.so`  |
| `ParallelFunctions.py` | Python | Wraps functions for the python MPI interface to be used for applying the particle load.                          | None    |
| `MakeGrid.pyx`         | Cython | Contains functions for grid deposition.                                | `MakeGrid.c` `MakeGrid.cpython-*.so`              |

Note that the compiled binaries provided will only work on a Linux platform with `x86-64` 64-bit architecture. If you are using a different set-up, please remove 
the binaries from your local clone and recompile the source. `MakeGrid.pyx` can be compiled with
```bash
cythonize -i MakeGrid.pyx
```
while for setting up `read_eagle.py`
we refer to the more detailed documentation in John Helly's [repository](https://github.com/jchelly/read_eagle). For a correct implementation of the `read_eagle.py`
module, it is important that at least the `.so` image binary and the `.py` source code are visible by the `PYTHONPATH` and located in the same directory.