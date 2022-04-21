import numpy as np
import yaml
from cosmology_functions import compute_masses, compute_softening


class ParticleLoadParams:
    def __init__(self, args):

        # These must be in the parameter file, others are optional.
        self.required_params = [
            "box_size",
            "n_particles",
            "glass_num",
            "f_name",
            "panphasian_descriptor",
            "ndim_fft_start",
            "is_zoom",
            "which_cosmology",
        ]

        # These cannot be in param file.
        self.arg_list = [
            "make_ic_gen_param_files",
            "make_swift_param_files",
            "save_pl_data",
            "save_pl_data_hdf5",
            "with_mpi",
        ]

        # Deal with command line arguments.
        self.read_args(args)

        # Read the parameter file.
        self.read_param_file()

        # Fill in default values.
        self.populate_defaults()

        # Sanity checks.
        self.sanity_checks()

        # Set cosmological parameters.
        self.set_cosmology()

        # Compute the mass of the volume given the cosmology.
        compute_masses(self)

        # Compute the softening lengths.
        compute_softening(self)

    def read_args(self, args):
        """Parse command line arguments."""

        # Parameter file.
        self.param_file = args.param_file

        print("\n")
        for att in self.arg_list:
            setattr(self, att, getattr(args, att))
            print(f"{att}: \033[92m{getattr(args, att)}\033[0m")
        print("\n")

    def sanity_checks(self):
        """Perform some basic sanity checks on params we have."""

        # Make sure coords is a numpy array.
        if not isinstance(self.coords, np.ndarray):
            self.coords = np.array(self.coords)

        assert (
            np.true_divide(self.n_particles, self.glass_num) % 1 < 1e-6
        ), "Number of particles must divide into glass_num"

        # Check we have the constraints descriptors.
        if self.num_constraint_files > 0:
            for i in range(self.num_constraint_files):
                assert hasattr(self, f"constraint_phase_descriptor{i+1}")
                assert hasattr(self, f"constraint_phase_descriptor{i+1}_path")
                assert hasattr(self, f"constraint_phase_descriptor{i+1}_levels")

        # Want to make ic files, have we said the path?
        if self.make_ic_gen_param_files:
            assert hasattr(self, "ic_dir")

        # Want to make SWIFT files, have we said the path?
        if self.make_swift_param_files:
            assert hasattr(self, "swift_dir")
            assert hasattr(self, "swift_ic_dir_loc")
            assert hasattr(self, "template_set")

        # For non-zoom simulations.
        if self.is_zoom == False:
            assert self.n_species == 1, "Must be 1, not a zoom"
            assert self.multigrid_ics == 0, "Must be 0, not a zoom" 
            assert hasattr(self, f"glass_file_loc")

    def set_cosmology(self):
        """Set cosmological params."""

        if self.which_cosmology == "Planck2013":
            self.Omega0 = 0.307
            self.OmegaCDM = 0.2587481
            self.OmegaLambda = 0.693
            self.OmegaBaryon = 0.0482519
            self.HubbleParam = 0.6777
            self.Sigma8 = 0.8288
            self.linear_ps = "extended_planck_linear_powspec"
        elif self.which_cosmology == "Planck2018":
            self.Omega0 = 0.3111
            self.OmegaLambda = 0.6889
            self.OmegaBaryon = 0.04897
            self.HubbleParam = 0.6766
            self.Sigma8 = 0.8102
            self.linear_ps = "EAGLE_XL_powspec_18-07-2019.txt"
        else:
            raise ValueError("Invalid cosmology")

    def read_param_file(self):
        """Read the particle load parameter YAML file."""

        with open(self.param_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # Make sure we have the minimum params.
        for att in self.required_params:
            assert att in data.keys(), f"Need to have {att} as required parameter."

        # Make sure these are not in param file.
        for att in self.arg_list:
            assert att not in data.keys(), f"Command line args cant be params"

        # Print and store read params.
        print(f"Loaded {len(data)} parameters from {self.param_file}:\n")
        for att in data.keys():
            print(f" - \033[96m{att}\033[0m: \033[92m{data[att]}\033[0m")
            setattr(self, att, data[att])
        print("\n")

    def populate_defaults(self):
        """Fill in the default values (won't overwrite passed params)."""

        self.add_default_value("coords", np.array([0.0, 0.0, 0.0]))
        self.add_default_value("radius", 0.0)
        self.add_default_value("mask_file", None)
        self.add_default_value("all_grid", False)
        self.add_default_value("n_species", 1)
        self.add_default_value("num_constraint_files", 0)
        self.add_default_value("nq_mass_reduce_factor", 1 / 2.0)
        self.add_default_value("skin_reduce_factor", 1 / 8.0)
        self.add_default_value("min_num_per_cell", 8)
        self.add_default_value("radius_factor", 1.0)
        self.add_default_value("glass_buffer_cells", 2)
        self.add_default_value("ic_region_buffer_frac", 1.0)
        self.add_default_value("starting_z", 127.0)
        self.add_default_value("finishing_z", 0.0)
        self.add_default_value("nmaxpart", 36045928)
        self.add_default_value("nmaxdisp", 791048437)
        self.add_default_value("mem_per_core", 18.2e9)
        self.add_default_value("max_particles_per_ic_file", 400**3)
        self.add_default_value("use_ph_ids", True)
        self.add_default_value("nbit", 21)
        self.add_default_value("fft_times_fac", 2.0)
        self.add_default_value("multigrid_ics", False)
        self.add_default_value("min_nq", 20)
        self.add_default_value("_max_nq", 1000)
        self.add_default_value("is_slab", False)
        self.add_default_value("grid_also_glass", False)
        self.add_default_value("glass_files_dir", "./glass_files/")
        self.add_default_value("softening_ratio_background", 0.02)
        self.add_default_value("ncores_node", 28)
        self.add_default_value("n_nodes_swift", 1)
        self.add_default_value("num_hours_swift", 10)
        self.add_default_value("swift_exec_location", ".")
        self.add_default_value("num_hours_ic_gen", 10)
        self.add_default_value("n_cores_ic_gen", 10)

    def add_default_value(self, att, value):
        """Add default parameter value to data array."""

        if not hasattr(self, att):
            setattr(self, att, value)
