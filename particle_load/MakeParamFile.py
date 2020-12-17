import re
import os
import subprocess
from string import Template

# |----------------------------------------|
# | Make submit and param file for IC gen. |
# |----------------------------------------|

def make_submit_file_ics(params):
    """ Make slurm submission script for icgen. """

    # Make folder if it doesn't exist.
    ic_gen_dir = '%s/ic_gen_submit_files/%s/'%(params['ic_dir'],params['f_name'])
    if not os.path.exists(ic_gen_dir): os.makedirs(ic_gen_dir)

    # Replace template values.
    with open('./templates/ic_gen/submit', 'r') as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open('%s/submit.sh'%(ic_gen_dir), 'w') as f:
        f.write(result)

    # Change execution privileges (make file executable by group)
    # Assumes the files already exist. If not, it has no effect.
    os.chmod(f"{ic_gen_dir}/submit.sh", 0o744)

def make_param_file_ics(params):
    """ Make parameter file for icgen. """

    # Make folder if it doesn't exist.
    ic_gen_dir = '%s/ic_gen_submit_files/%s/'%(params['ic_dir'],params['f_name'])
    if not os.path.exists(ic_gen_dir): os.makedirs(ic_gen_dir) 

    # Make output folder for the Ics.
    ic_gen_output_dir = '%s/ic_gen_submit_files/%s/ICs/'%(params['ic_dir'],params['f_name'])
    if not os.path.exists(ic_gen_output_dir): os.makedirs(ic_gen_output_dir)

    # Minimum FFT grid that fits 2x the nyquist frequency.
    ndim_fft = params['ndim_fft_start']
    N = (params['high_res_n_eff'])**(1./3) if params['is_zoom'] else (params['n_particles'])**(1/3.) 
    while float(ndim_fft)/float(N) < params['fft_times_fac']:
        ndim_fft *= 2
    print("Using ndim_fft = %d" % ndim_fft)
    params['ndim_fft'] = ndim_fft

    # What are the phase descriptors?
    if params['constraint_phase_descriptor'] != '%dummy':
        if params['constraint_phase_descriptor2'] != '%dummy':
            params['is_constraint'] = 2
        else:
            params['is_constraint'] = 1
    else:
        params['is_constraint'] = 0
    params['constraint_path'] = '%dummy' if params['constraint_phase_descriptor'] == '%dummy' else\
            "'%s'"%params['constraint_phase_descriptor_path']
    params['constraint_path2'] = '%dummy' if params['constraint_phase_descriptor2'] == '%dummy' else\
            "'%s'"%params['constraint_phase_descriptor_path2']

    # Is this a zoom simulation (zoom can't use 2LPT)?
    if params['is_zoom']:
        if params['is_slab']:
            params['two_lpt'] = 1
            params['multigrid'] = 0
        else:
            params['two_lpt'] = 0 if params['multigrid_ics'] else 1
            params['multigrid'] = 1 if params['multigrid_ics'] else 0
    else:
        params['high_res_L'] = 0.0
        params['high_res_n_eff'] = 0
        params['two_lpt'] = 1 
        params['multigrid'] = 0

    # Use peano hilbert indexing?
    params['use_ph'] = 2 if params['use_ph_ids'] else 1

    # Cube of neff
    params['high_res_n_eff_cube'] = round(params['high_res_n_eff']**(1./3))

    # Replace template values.
    with open('./templates/ic_gen/params.inp', 'r') as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open('%s/params.inp'%(ic_gen_dir), 'w') as f:
        f.write(result)

# |-------------------------------------------|
# | Make submit and parameter file for SWIFT. |
# |-------------------------------------------|

def make_param_file_swift(params):

    # Make data dir.
    data_dir = params['swift_dir'] + '%s/'%params['f_name']
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(data_dir + 'out_files/'): os.makedirs(data_dir + 'out_files/')

    # Starting and finishing scale factors.
    params['starting_a'] = 1./(1+float(params['starting_z']))
    params['finishing_a'] = 1./(1+float(params['finishing_z']))

    # Replace values.
    if 'tabula_' in params['template_set'].lower():
        raise Exception("Fix this one")
        #r = ['%.5f'%h, '%.8f'%starting_a, '%.8f'%finishing_a, '%.8f'%omega0, '%.8f'%omegaL,
        #'%.8f'%omegaB, fname, fname, '%.8f'%(eps_dm/h),
        #'%.8f'%(eps_baryon/h), '%.3f'%(softening_ratio_background),
        #'%.8f'%(eps_baryon_physical/h), '%.8f'%(eps_dm_physical/h), fname]

        #subprocess.call("cp ./templates/swift/%s/select_output.yml %s"%\
        #        (template_set, data_dir), shell=True)
    elif params['template_set'].lower() == 'sibelius':
        # Copy over select output.
        subprocess.call("cp ./templates/swift/%s/select_output.yml %s"%\
                (params['template_set'], data_dir), shell=True)

        # Copy over snapshots times.
        subprocess.call("cp ./templates/swift/%s/stf_times_a.txt %s/snapshot_times.txt"%\
                (params['template_set'], data_dir), shell=True)
    elif params['template_set'].lower() == 'eaglexl':
        raise Exception("Fix this one")
        #split_mass = gas_particle_mass / 10**10. * 4.
        #r = [fname, '%.5f'%h, '%.8f'%starting_a, '%.8f'%finishing_a, '%.8f'%omega0, '%.8f'%omegaL,
        #'%.8f'%omegaB, fname, '%.8f'%(eps_dm/h),
        #'%.8f'%(eps_baryon/h), '%.8f'%(eps_dm_physical/h),
        #'%.8f'%(eps_baryon_physical/h), '%.3f'%(softening_ratio_background), 
        #'%.8f'%split_mass, ic_dir, fname]
    else:
        raise ValueError("Invalid template set")

    t_file = './templates/swift/%s/params.yml'%params['template_set']

    # Replace template values.
    with open(t_file, 'r') as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open('%s/params.yml'%(data_dir), 'w') as f:
        f.write(result)

def make_submit_file_swift(params):

    data_dir = params['swift_dir'] + '%s/'%params['f_name']
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    s_file = './templates/swift/%s/submit'%params['template_set'].lower()
    rs_file = './templates/swift/%s/resubmit'%params['template_set'].lower()

    # Replace template values.
    with open(s_file, 'r') as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open('%s/submit'%(data_dir), 'w') as f:
        f.write(result)

    # Replace template values.
    with open(rs_file, 'r') as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open('%s/resubmit'%(data_dir), 'w') as f:
        f.write(result)

    with open('%s/auto_resubmit'%data_dir, 'w') as f:
        f.write('sbatch resubmit')

    # Change execution privileges (make files executable by group)
    # Assumes the files already exist. If not, it has no effect.
    os.chmod(f"{data_dir}/submit", 0o744)
    os.chmod(f"{data_dir}/resubmit", 0o744)
    os.chmod(f"{data_dir}/auto_resubmit", 0o744)
