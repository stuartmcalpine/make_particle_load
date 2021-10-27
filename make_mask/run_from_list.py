"""
Script to batch process multiple haloes from one parent simulation.

Run as:
```
python3 run_from_list.py \
   --template [name of parameter template file] \
   --listfile [name of file containing the group numbers to process]
```
"""
import argparse
import os
import re
from shutil import copyfile
from numpy import loadtxt, ndarray
from yaml import safe_load as load
from typing import Tuple

from make_mask import MakeMask


def make_masks_from_list() -> None:
    """Main wrapper function to generate the masks."""

    parser = argparse.ArgumentParser(
        description="Script to batch process multiple haloes from the same "
                    "parent simulation."
    )
    parser.add_argument('-t', '--template_file',
        help="The name of the template parameter file to generate the mask, "
             "relative to the directory containing this script.")
    parser.add_argument('-l', '--list_file',
        help="The name of the file containing the group numbers to process, "
             "relative to the directory containing this script.")
    args = parser.parse_args()

    # Parse the parameter (template), and check the expected placeholders
    params = load(open(args.template_file))
    params['select_from_vr'] = True
    if 'GROUPNUMBER' not in params['fname']:
        raise ValueError(
            "'fname' parameter in the template file must contain the "
            f"placeholder 'GROUPNUMBER', but is '{params['fname']}'!"
        )

    # Copy the template parameter file to the output directory    
    out_dir = get_output_dir_from_template()    # out_dir is created if needed
    this_file_directory = os.path.dirname(__file__)
    copyfile(
        os.path.join(this_file_directory, args.template_file),
        os.path.join(out_dir, os.path.basename(args.template_file))
    )

    group_numbers, sorter = get_group_numbers_list()
    if sorter is not None and 'sort_type' in params:
        if sorter.lower() != params['sort_type'].lower():
            print(
                f"Warning: list file appears to refer to sort_type "
                f"'{sorter}', but parameter file specifies sort_type "
                f"'{params['sort_type']}'. Overriding parameter file..."
            )

    for group in group_numbers:
        params_grp = copy.deepcopy(params)
        params_grp['fname'].replace('GROUPNUMBER', str(group_number))
        params_grp['group_number'] = group_number
        mask = MakeMask(params=params_grp)


def get_output_dir_from_template() -> str:
    params = load(open(args.template))
    output_dir = params['output_dir']
    if not os.path.isdir(output_dir):
        os.path.makedirs(output_dir)
    return output_dir


def get_group_numbers_list(list_file: str) -> Tuple[np.ndarray, str]:
    # Check whether the first line contains a header indicating the type
    # of sorter (M200c, M500c)
    with open(list_file) as f:
        header = f.readline()
        if header[0] == '#':
            if 'm200crit' in header.lower():
                sorter = 'm200crit'
            elif 'm500crit' in header.lower():
                sorter = 'm500crit'
            elif 'none' in header.lower():
                sorter = 'none'
            else:
                sorter = None

    return loadtxt(list_file).astype(int), sorter


if __name__ == '__main__':
    make_masks_from_list()
