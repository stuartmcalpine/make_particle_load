import argparse
import os
import re
from shutil import copyfile
from numpy import loadtxt, ndarray
from yaml import load

from Generate_PL import ParticleLoad

this_file_directory = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    description=(
        "Generates template and submission files for running depositing the particle load."
    ),
    epilog=(
        "Example usage: "
        "python3 submit_from_list.py "
        "-t param_files/default_sept.yml "
        "-l /cosma7/data/dp004/dc-alta2/xl-zooms/ics/masks/groupnumbers_defaultSept.txt "
        "-s "
    ),
)

parser.add_argument(
    '-t'
    '--template',
    action='store',
    required=True,
    type=str
)

parser.add_argument(
    '-l'
    '--listfile',
    action='store',
    required=True,
    type=str
)

parser.add_argument(
    '--only-calc-ntot',
    action='store_true',
    default=False,
    required=False,
)

parser.add_argument(
    '-s'
    '--submit',
    action='store_true',
    default=False,
    required=False,
)

args = parser.parse_args()


def get_output_dir_from_template() -> str:
    params = load(open(args.template))
    output_dir = params['output_dir']
    if not os.path.isdir(output_dir):
        raise OSError(f"The specified output directory does not exist. Trying to save to {output_dir}")
    return output_dir


def get_run_name_from_template() -> str:
    params = load(open(args.template))
    return params['fname']


def get_group_numbers_list() -> ndarray:
    return loadtxt(args.listfile).astype(int)


def get_mass_sort_key() -> str:
    with open(args.listfile, "r") as selection:
        lines = selection.readlines()
        for line in lines:
            if 'mass_sort' in line:
                sort_key = line.split()[-1]
                break
    assert '500' in sort_key or '200' in sort_key, ("Mass sort key returned unexpected value.",
                                                    f"Expected `M_200crit` or `M_500crit`, got {sort_key}")
    return sort_key


def replace_pattern(pattern: str, replacement: str, filepath: str):
    with open(filepath, "r") as sources:
        lines = sources.readlines()
    with open(filepath, "w") as sources:
        for line in lines:
            sources.write(re.sub(pattern, replacement, line))


def make_masks_from_list() -> None:
    out_dir = get_output_dir_from_template()
    copyfile(
        os.path.join(this_file_directory, args.template),
        os.path.join(out_dir, os.path.basename(args.template))
    )

    for group_number in get_group_numbers_list():

        mask_name = get_run_name_from_template().replace('GROUPNUMBER', str(group_number))
        mask_paramfile = os.path.join(out_dir, f"{mask_name}.yml")
        copyfile(os.path.join(out_dir, os.path.basename(args.template)), mask_paramfile)
        replace_pattern('GROUPNUMBER', str(group_number), mask_paramfile)
        sort_key = get_mass_sort_key()
        if '500' in sort_key:
            replace_pattern('SORTM200', '0', mask_paramfile)
            replace_pattern('SORTM500', '1', mask_paramfile)
        elif '200' in sort_key:
            replace_pattern('SORTM200', '1', mask_paramfile)
            replace_pattern('SORTM500', '0', mask_paramfile)

        ParticleLoad(mask_paramfile, only_calc_ntot=args.only_calc_ntot)


if __name__ == '__main__':
    make_masks_from_list()