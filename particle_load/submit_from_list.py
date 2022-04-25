import argparse
import os
import re
import sys
from shutil import copyfile
from typing import List

from yaml import load

this_file_directory = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    description=(
        "Generates template and submission files for running depositing the particle load."
    ),
    epilog=(
        "Example usage: "
        "python3 submit_from_list.py "
        "-t /cosma/home/dp004/dc-alta2/data7/xl-zooms/ics/particle_loads/template_-8res.yml"
        "-l /cosma/home/dp004/dc-alta2/data7/xl-zooms/ics/particle_loads/masks_list.txt "
        "-p /cosma/home/dp004/dc-alta2/make_particle_load/particle_load/Generate_PL.py "
        "-n 28 "
        "-s "
    ),
)

parser.add_argument(
    "-t",
    "--template",
    action="store",
    required=True,
    type=str,
    help=(
        "The master parameter file to use as a template to generate mask-specific ones. "
        "It usually contains the hot keywords PATH_TO_MASK and FILENAME, which can be replaced "
        "with mask-dependent values."
    ),
)

parser.add_argument(
    "-x",
    "--template-slurm",
    action="store",
    required=True,
    type=str,
    help=(
        "The master SLURM submission file to use as a template to generate object-specific ones. "
        "It usually contains the hot keywords N_TASKS and RUN_NAME, which can be replaced "
        "with object-dependent values."
    ),
)

parser.add_argument(
    "-l",
    "--listfile",
    action="store",
    required=True,
    type=str,
    help=(
        "The file with the list of full paths to the mask files that are to be handled. "
        "The file paths are required to end with the file name with the correct `.hdf5` extension. "
        "The base-name of the masks files is used to replace the hot keywords PATH_TO_MASK and "
        "FILENAME in the template."
    ),
)

parser.add_argument(
    "-s",
    "--submit",
    action="store_true",
    default=False,
    required=False,
    help=(
        "If activated, the program automatically executes the command `sbatch submit.sh` and launches the "
        "`IC_Gen.x` code for generating initial conditions. NOTE: all particle load in the list will be submitted "
        "to the SLURM batch system as individual jobs."
    ),
)

parser.add_argument(
    "-p",
    "--particle-load-library",
    action="store",
    type=str,
    default=os.path.join(os.getcwd(), "Generate_PL.py"),
    required=False,
    help=(
        "If this script is not located in the same directory as the `Generate_PL.py` code, you can import "
        "the code as an external library by specifying the full path to the `Generate_PL.py` file."
    ),
)

parser.add_argument(
    "-n",
    "--ntasks",
    action="store",
    default=28,
    type=int,
    required=False,
    help=(
        "Number of MPI ranks to use for the generation of the particle load. NOTE: the particle load stored in "
        "binary Fortran files is split into as many files as there are ranks. `IC_Gen.x` cannot allocate more "
        "than 400^3 particles per MPI rank, therefore make sure to use enough MPI ranks such that each binary "
        "file contains less that 400^3 = 64M particles."
    ),
)

parser.add_argument(
    "-d",
    "--dry",
    action="store_true",
    default=False,
    required=False,
    help=(
        "Use this option to produce dry runs, where the `ParticleLoad` class is deactivated, as well as the "
        "functionality for submitting jobs to the queue automatically, i.e. overrides --submit. Use this for "
        "testing purposes."
    ),
)

args = parser.parse_args()

try:
    import Generate_PL
except ImportError:
    pass
else:
    sys.path.append(os.path.split(args.particle_load_library)[0])
    try:
        import Generate_PL
    except ImportError:
        raise Exception(
            "Make sure you have added the `Generate_PL.py` module directory to your $PYTHONPATH."
        )


def print_parser_args() -> None:
    print(f"{' PARSER ARGS ':-^60}")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print(f"\n{' FILES ':-^60}")


def get_mask_paths_list() -> List[str]:
    with open(args.listfile) as f:
        lines = f.read().splitlines()

    group_numbers = []

    for line in lines:
        assert line.endswith(
            ".hdf5"
        ), f"Extension of the mask file {line} is ambiguous. File path must end in `.hdf5`."

        mask_name = os.path.splitext(os.path.split(line)[-1])[0]
        group_numbers.append(int(mask_name.split("_VR")[-1]))

    lines_sorted = [x for _, x in sorted(zip(group_numbers, lines))]
    return lines_sorted


def get_output_directory() -> str:
    return os.path.split(args.listfile)[0]


def replace_pattern(pattern: str, replacement: str, filepath: str):
    with open(filepath, "r") as sources:
        lines = sources.readlines()
    with open(filepath, "w") as sources:
        for line in lines:
            sources.write(re.sub(pattern, replacement, line))


def get_from_template(parameter: str) -> str:
    params = load(open(args.template))
    return params[parameter]


def make_particle_load_from_list() -> None:
    out_dir = get_output_directory()
    if not os.path.isfile(os.path.join(out_dir, os.path.basename(args.template))):
        copyfile(
            os.path.join(this_file_directory, args.template),
            os.path.join(out_dir, os.path.basename(args.template)),
        )
    if not os.path.isfile(os.path.join(out_dir, os.path.basename(args.template_slurm))):
        copyfile(
            os.path.join(this_file_directory, args.template_slurm),
            os.path.join(out_dir, os.path.basename(args.template_slurm)),
        )
    if not os.path.isdir(os.path.join(out_dir, "logs")):
        os.mkdir(os.path.join(out_dir, "logs"))

    sbatch_calls = []

    for mask_filepath in get_mask_paths_list():

        # Construct particle load parameter file name
        mask_name = os.path.splitext(os.path.split(mask_filepath)[-1])[0]

        file_name = get_from_template("f_name").replace("FILENAME", str(mask_name))

        # Edit parameter file
        particle_load_paramfile = os.path.join(out_dir, f"{file_name}.yml")
        copyfile(
            os.path.join(out_dir, os.path.basename(args.template)),
            particle_load_paramfile,
        )
        replace_pattern("PATH_TO_MASK", str(mask_filepath), particle_load_paramfile)
        replace_pattern("FILENAME", str(mask_name), particle_load_paramfile)

        # Edit SLURM submission file
        particle_load_submit = os.path.join(out_dir, f"{file_name}.slurm")
        copyfile(
            os.path.join(out_dir, os.path.basename(args.template_slurm)),
            particle_load_submit,
        )
        replace_pattern("N_TASKS", str(args.ntasks), particle_load_submit)
        replace_pattern("RUN_NAME", f"{file_name}", particle_load_submit)
        replace_pattern(
            "PL_INVOKE",
            (
                f"cd {os.path.split(args.particle_load_library)[0]}\n"
                f"mpirun -np $SLURM_NTASKS python3 Generate_PL.py {particle_load_paramfile}"
            ),
            particle_load_submit,
        )
        print(f"ParticleLoad({particle_load_paramfile})")

        if args.submit:
            # Write IC_Gen.x submission command upon PL completion
            ic_submit_dir = os.path.join(
                get_from_template("ic_dir"), "ic_gen_submit_files", file_name
            )
            replace_pattern(
                "ICGEN_SUBMIT",
                f"cd {ic_submit_dir}\nsbatch submit.sh",
                particle_load_submit,
            )
            print(f"Submitting IC_Gen.x at {ic_submit_dir}")
        else:
            replace_pattern("ICGEN_SUBMIT", "", particle_load_submit)

        # Change execution privileges (make files executable by group)
        # Assumes the files already exist. If not, it has no effect.
        os.chmod(f"{particle_load_submit}", 0o744)

        sbatch_calls.append(
            r"sbatch {0:s}".format(os.path.basename(particle_load_submit))
        )

        # if not args.dry:
        #     old_cwd = os.getcwd()
        #     os.chdir(out_dir)
        #     print(f"\nCalling:\ncd {os.getcwd()}\nsbatch {os.path.basename(particle_load_submit)}")
        #     os.system(r"sbatch {0:s}".format(os.path.basename(particle_load_submit)))
        #     os.chdir(old_cwd)

    for i in sbatch_calls:
        print(i)


if __name__ == "__main__":
    print_parser_args()
    make_particle_load_from_list()
