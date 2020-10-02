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
to update to the latest changes. The masking and particle-load scripts rely on two groups of dependencies: the ones 
that are specific to this project are collected in the `modules` sub-directory and some external ones, which can be 
installed and correctly configured following the instructions below.

Internal dependencies
--------
The `modules` directory contains the following project-specific dependencies:

| Package name           | Platform   | Utility                                                                | Compiled binaries                                 |
|:---------------------- |------------|------------------------------------------------------------------------|---------------------------------------------------|
| `peano.py`             | Python     | Translates particle IDs into coordinates from the initial conditions.  | None                                              |
| `read_swift.py`        | Python     | Reads snapshots generated using the SWIFT code.                        | None                                              |
| `ParallelFunctions.py` | Python     | Wraps functions for the python MPI interface to be used for applying the particle load.                          | None    |
| `MakeGrid.pyx`         | Cython     | Contains functions for grid deposition.                                | `MakeGrid.c` `MakeGrid.cpython-*.so`              |

The `peano.py` and `read_swift.py` packages are automatically added to the `PYTHONPATH` environment variable at runtime
whenever the scripts using them are invoked. In the case of `peano.py`, this is achieved using
```python
import sys

sys.path.append("path/to/modules/directory/containing/peano")

try:
    import peano
except ImportError:
    raise Exception("Make sure you have added the `peano.py` module directory to your $PYTHONPATH.")
except:
    raise Exception(
        "Something else has gone wrong with importing the `peano.py` module. "
        "Check the path appended to the $PYTHONPATH points to the correct directory, "
        "that `peano.py` is set-up correctly and all its dependencies are in correct working order."
    )
```
For this project, scripts will invoke internal dependencies dynamically using the syntax above. If you need to export any
of the internal dependencies to your script, you can also call `sys.path.append()` to configure them for your custom 
application.

As part of the particle load production, the `MakeGrid.pyx` is written in Cython and need to be compiled prior any use.
you can translate `MakeGrid.pyx` into native C code and compile it into a binary image using
```bash
cythonize -i MakeGrid.pyx
```
run in the same directory as `MakeGrid.pyx`. This operation produces the `MakeGrid.c` file, which encloses a C-translation
of the original Cython syntax and the `MakeGrid.cpython-*.so` shared object, which is the file that the Python interpreter
will actually point to when the `import` command is called. You may remove `MakeGrid.c` as it is only used by the Cython compiler 
to generate the shared object and is not needed by any other scripts at runtime.

External dependencies
--------
To allow back-compatibility with EAGLE-type simulations and other datasets produced with Gadget-2/3, we use the `read_eagle.py`
module, developed by John Helly and hosted in their [GitHub repository](https://github.com/jchelly/read_eagle). You can 
set-up this external module by cloning it into a directory of your preference (e.g. could be your `/home` directory) using
```bash
git clone https://github.com/jchelly/read_eagle.git
```
If you are using DiRAC's Cosma clusters, then import the following modules
```bash
module load cosma, python/3.6.5, hdf5, intel_mpi
```
otherwise make sure you have the equivalent libraries installed in your system.
After that, navigate into the newly created `read_eagle` directory, where a `setup.py` file is located, and run
```bash
python3 ./setup.py install --prefix=/path/to/install/dir/
```
where you can indicate the directory in which the libraries are to be installed in the `--prefix` argument. If you are
planning on using the `read_eagle.py` module only for this project, we recommend installing it in the `/modules` directory.
This command will create the path `/lib/python3.6/site-packages` in the install directory and such path points to three 
files (values indicated inside `{...}` may change depending on your Python version or OS platform): 
- `read_eagle.py`, which contains Python wrappers around the C library,
- `ReadEagle-{1.0}-py{3.6}.egg-info`, which contains metadata about the package version and the target 
Python version detected by the compiler,
- `_read_eagle.cpython-{36m}-{x86_64-linux-gnu}.so` is the shared object file containing the C-compiled library invoked by 
`read_eagle.py`.

Given the format of the `read_eagle` Python wrapper, it is important that `read_eagle.py` and `_read_eagle.cpython-{36m}-{x86_64-linux-gnu}.so`
are located in the same directory for a correct implementation. In order to call `import read_eagle` from an external 
project, such as the zooms initial conditions pipeline, you need to append the `read_eagle.py` path to `$PYTHONPATH`.
If you installed `read_eagle` in `/modules`, we recommend moving `read_eagle.py` and `_read_eagle.cpython-{36m}-{x86_64-linux-gnu}.so`
three directories up, at the same level as the project-specific modules (at this point, 
you may delete the `/modules/lib/python3.6/site-packages` path). When appending the `/modules` path to `$PYTHONPATH` via 
`sys.path.append()`, the interpreter will also be able to import `read_eagle.py`.

Debug tips
-------
This section contains some useful debugging tips related to dependency configurations.

In order to check that you have the correct `PYTHONPATH` environment variable set-up and that all your dependencies are 
visible to the Python interpreter, you can print your current `PYTHONPATH` using
```python
import sys
print(sys.path)
```
The output contains a list of the directories, including the ones you have appended with `sys.path.append()`. We 
discourage the use of `os.environ['PYTHONPATH']`, as it may produce OS platform-dependent outputs, while its `sys` 
equivalent is platform-independent. If your custom directory appears in the `sys.path`, but the code exits with an
`ImportError`, check that the module is placed in the correct directory.