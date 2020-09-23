Introduction
----------
This directory contains two `python3` scripts for generating masks
for zoom simulation initial conditions and a folder with example parameter 
files to be parsed into the scripts. The scripts allow to generate an individual 
mask, using `make_mask.py`, or to generate a masks for a sample of objects
in a loop using `run_from_list.py`.

Masking individual objects
----------
If you intend to run a single mask, you may run the `make_mask.py` file as 
```shell script
python3 make_mask.py param_file.yml
```
where `test.yml` is the parameter file with the details of the parent box,
optional Velociraptor catalogues and inputs controlling the mask topology.
Note that for high-resolution parent boxes, or generally for cases where the
mask is generated from a large number of particles, `make_mask.py` can also
be run on parallel MPI ranks with
```shell script
mpirun -np $NUM_CPUS python3 make_mask.py param_file.yml
```
This option allows to perform the 3-dimensional binning of the particles in 
the parent box's initial conditions using `mpi4py`; the other functions, 
such as volume extrusion are implemented serially, since they operate on the 
already binned coordinates and do not affect the runtime for most applications.

Masking object samples
----------
If you require to iterate the masking procedure over many objects, it might
be useful to automate the `python3 make_mask.py param_file.yml` command.
For this purpose, you can use the `run_from_list.py` scripts, supplied with a
groups list and a parameter file template. In order to reduce the level of 
interaction with the scripts, we recommend configuring the parameter files
in automatic mode, by specifying a Velociraptor output, a sorting rule and a 
`{group-number-keyword}` in the parameter file template.
Similarly to the individual masks, the script can be run in full-serial mode with
```shell script
python3 run_from_list.py \
  --template param_file_template.yml \
  --listfile groupnumbers_list.txt
```
or with MPI parallelism on each individual mask iteration
```shell script
mpirun -np $NUM_CPUS python3 run_from_list.py \
  --template param_file_template.yml \
  --listfile groupnumbers_list.txt
```

Outputs
----------
Every time the `MakeMask` class is invoked from the `make_mask.py` file, two outputs are produced
in the `out_dir` specified in the parameter file: a plot of the three x-, y-, z-projections of the
mask with the over-plotted initial position of the particles and an `hdf5` file containing the 
coordinates and bin-size of the mask's sampling points that will be used my the particle-load 
script. The `upload-mask` branch also parses the items in the parameter file as attributes for
the `hdf5` file, in order to keep a record of the inputs even in the case where parameter files
are unsaved or lost. You can access these attributes using `h5py`'s `File` instance by typing 
`h5file.attrs.items()`.

Example: parameter file template
----------
The `run_from_list.py` script accepts a parameter file template, which will then
copied into the output directory and automatically edited with the parameters
relative to the specific object to mask. For this reason, it is important that
mutable values, such as the object's index in the Velociraptor catalogue (group number),
are clearly labelled with recognisable keywords in the parameter file template.
The current convention includes the following:
- `SORTM200 (bool)` is replaced with 0 or 1 depending on the sorting scheme of group numbers list,
- `SORTM500 (bool)` is replaced with 0 or 1 depending on the sorting scheme of group numbers list,
- `GROUPNUMBER (int)` is replaced with the integer index for each object in the group numbers list.

You may edit these keywords by consistently adapting the `run_from_list.py` script. For the sake of 
clarity, the current version of the template follows a convention for defining the run name:
`{name-of-parent-box}_VR{group-number}` (e.g. `L0300N0564_VR10` is the name of the 10<sup>th</sup> 
Velociraptor object drawn from the `L0300N0564` cosmological box). Below is reported an example of 
a parameter file template used for a group sample in EAGLE-XL.
```yaml
# SET-UP MASK #
fname:             L0300N0564_VRGROUPNUMBER    # The save name for the mask file
snap_file:         /cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/EAGLE-XL_L0300N0564_DMONLY_0036.hdf5  # The location of the snapshot we are creating the mask from
bits:              21                # The number of bits used in the particle IDs for the Peano-Hilbert indexing (EAGLE runs use 14)
shape:             sphere            # Shape of the region to reproduce. Available are: 'sphere', 'cubiod', 'slab'
data_type:         swift             # Can be 'gadget' or 'swift' (default 'swift')
divide_ids_by_two: False             # True if you need to divide PIDs by two to get back to ICS (needed for eagle)
min_num_per_cell:  3                 # Minimum particles per cell (default 3). Cells with less than `min_num_per_cell` particles are ignored
mpc_cell_size:     1.2               # Cell size in Mpc / h (default 3.)
select_from_vr:    1                 # If set to 1, it enables automatic groups search from the Velociraptor catalogue. Set to 0 for manual selection

# AMOEBA TOPOLOGY #
topology_fill_holes:       1         # Toggle algorithm for filling holes. Set value to 1 to enable, 0 to disable (default 0)
topology_dilation_niter:   2         # Number of iterations of the algorithm for extrusion. Set value to 0 to disable (default 0)
topology_closing_niter:    2         # Number of iterations of the algorithm for rounding edges. Set value to 0 to disable (default 0)

# AUTOMATIC GROUP SELECTION #
vr_file:             /cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/stf_swiftdm_3dfof_subhalo_0036/stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0 # The location of the Velociraptor group catalogue (to find coordinates of given GN)
sort_m200crit:       SORTM200        # If 1, sorts groups in the VR catalogue by M_200crit. Overrides `sort_m500crit` if both specified
sort_m500crit:       SORTM500        # If 1, sorts groups in the VR catalogue by M_500crit
GN:                  GROUPNUMBER     # The Group-Number of the halo in the structure-finding catalogue (requires `vr_file`). GN relative to the sorting rule
#highres_radius_r200: 5.             # How many times r200 of the passed group do you want to re-simulate. Overrides `highres_radius_r500` if both selected
highres_radius_r500: 10.             # How many times r500 of the passed group do you want to re-simulate

# OUTPUT DIRECTORY #
output_dir:           /cosma7/data/dp004/dc-alta2/xl-zooms/ics/masks # Directory where the mask image and hdf5 output are saved
```

Example: group numbers list
----------
For iterating the masking procedure over many groups, `run_from_list.py` needs to be pointed to a
text file containing the list of integers representing the group number of the objects in the Velociraptor 
catalogue, as well as optional information, such as the mass-sorting key. The layout of the list file does not
follow strict rules and may be adapted by editing the `run_from_list.py` code itself. In the context of EAGLE-XL,
we use the following structure for supplying the `groupnumbers_list.txt` file.
```text
# mass_sort: M_500crit
# Halo index:
3573
3032
2928
2915
2414
1857
1564
1236
1191
924
868
801
...
```
In this case, `mass_sort` can assume the values of `M_500crit` or `M_200crit`, while the parameter file template will 
automatically be edited to accommodate this choice for each mask iteration.