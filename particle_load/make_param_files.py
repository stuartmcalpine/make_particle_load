import os
import re
import subprocess
from string import Template

import numpy as np


def build_param_dict(pl_params, high_res_region):

    # Compute mass cut offs between particle types.
    if pl_params.is_zoom:
        if pl_params.n_species >= 2:
            hr_cut = np.log10(high_res_region.cell_info["glass_mass"]) + 0.01
            print("log10 mass cut from parttype 1 --> 2 = %.2f" % (hr_cut))
        if pl_params.n_species == 3:
            lr_cut = np.log10(high_res_region.cell_info["max_grid_mass"]) + 0.01
            print("log10 mass cut from parttype 2 --> 3 = %.2f" % (lr_cut))
    else:
        hr_cut = 0.0
        lr_cut = 0.0
        pl_params.high_res_L = 0.0
        pl_params.high_res_n_eff = 0

    # Glass files for non zooms.
    if pl_params.is_zoom:
        import_file = 0
        pl_basename = "./ic_gen_submit_files/$f_name/particle_load/fbinary/PL"
        pl_rep_factor = 1
    else:
        import_file = -1
        pl_basename = pl_params.glass_file_loc
        pl_rep_factor = int(
            np.rint((pl_params.n_particles / pl_params.glass_num) ** (1 / 3.0))
        )
        assert (
            pl_rep_factor**3 * pl_params.glass_num == pl_params.n_particles
        ), "Error rep_factor"

    # Build parameter list.
    param_dict = dict(
        import_file=import_file,
        pl_basename=pl_basename,
        pl_rep_factor=pl_rep_factor,
        hr_cut="%.3f" % hr_cut,
        lr_cut="%.3f" % lr_cut,
        is_zoom=int(pl_params.is_zoom),
        f_name=pl_params.f_name,
        n_species=pl_params.n_species,
        ic_dir=pl_params.ic_dir,
        box_size="%.8f" % pl_params.box_size,
        starting_z="%.8f" % pl_params.starting_z,
        finishing_z="%.8f" % pl_params.finishing_z,
        n_particles=pl_params.n_particles,
        coords_x="%.8f" % pl_params.coords[0],
        coords_y="%.8f" % pl_params.coords[1],
        coords_z="%.8f" % pl_params.coords[2],
        high_res_L="%.8f" % pl_params.high_res_L,
        high_res_n_eff=pl_params.high_res_n_eff,
        panphasian_descriptor=pl_params.panphasian_descriptor,
        ndim_fft_start=pl_params.ndim_fft_start,
        Omega0="%.8f" % pl_params.Omega0,
        OmegaCDM="%.8f" % pl_params.OmegaCDM,
        OmegaLambda="%.8f" % pl_params.OmegaLambda,
        OmegaBaryon="%.8f" % pl_params.OmegaBaryon,
        HubbleParam="%.8f" % pl_params.HubbleParam,
        Sigma8="%.8f" % pl_params.Sigma8,
        is_slab=pl_params.is_slab,
        use_ph_ids=pl_params.use_ph_ids,
        multigrid_ics=pl_params.multigrid_ics,
        linear_ps=pl_params.linear_ps,
        nbit=pl_params.nbit,
        fft_times_fac=pl_params.fft_times_fac,
        swift_ic_dir_loc=pl_params.swift_ic_dir_loc,
        softening_ratio_background=pl_params.softening_ratio_background,
        template_set=pl_params.template_set,
        gas_particle_mass=pl_params.gas_particle_mass,
        swift_dir=pl_params.swift_dir,
        n_nodes_swift="%i" % pl_params.n_nodes_swift,
        num_hours_swift=pl_params.num_hours_swift,
        swift_exec_location=pl_params.swift_exec_location,
        num_hours_ic_gen=pl_params.num_hours_ic_gen,
        n_cores_ic_gen="%i" % pl_params.n_cores_ic_gen,
        eps_dm="%.8f" % (pl_params.eps_dm / pl_params.HubbleParam),
        eps_baryon="%.8f" % (pl_params.eps_baryon / pl_params.HubbleParam),
        eps_dm_physical="%.8f" % (pl_params.eps_dm_physical / pl_params.HubbleParam),
        eps_baryon_physical="%.8f"
        % (pl_params.eps_baryon_physical / pl_params.HubbleParam),
        num_constraints=pl_params.num_constraint_files,
    )

    for i in range(pl_params.num_constraint_files):
        param_dict[f"constraint_phase_descriptor{i+1}"] = getattr(
            pl_params, f"constraint_phase_descriptor{i+1}"
        )
        param_dict[f"constraint_phase_descriptor{i+1}_path"] = getattr(
            pl_params, f"constraint_phase_descriptor{i+1}_path"
        )
        param_dict[f"constraint_phase_descriptor{i+1}_levels"] = getattr(
            pl_params, f"constraint_phase_descriptor{i+1}_levels"
        )

    return param_dict


# |----------------------------------------|
# | Make submit and param file for IC gen. |
# |----------------------------------------|


def make_submit_file_ics(params):
    """Make slurm submission script for icgen."""

    # Make folder if it doesn't exist.
    ic_gen_dir = "%s/%s/" % (params["ic_dir"], params["f_name"])
    if not os.path.exists(ic_gen_dir):
        os.makedirs(ic_gen_dir)

    # Replace template values.
    with open("./templates/ic_gen/submit", "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/submit.sh" % (ic_gen_dir), "w") as f:
        f.write(result)

    # Change execution privileges (make file executable by group)
    # Assumes the files already exist. If not, it has no effect.
    os.chmod(f"{ic_gen_dir}/submit.sh", 0o744)


def make_param_file_ics(params):
    """Make parameter file for icgen."""

    # Make folder if it doesn't exist.
    ic_gen_dir = "%s/%s/" % (params["ic_dir"], params["f_name"])
    if not os.path.exists(ic_gen_dir):
        os.makedirs(ic_gen_dir)

    # Make output folder for the Ics.
    ic_gen_output_dir = "%s/%s/ICs/" % (
        params["ic_dir"],
        params["f_name"],
    )
    if not os.path.exists(ic_gen_output_dir):
        os.makedirs(ic_gen_output_dir)

    # Minimum FFT grid that fits 2x the nyquist frequency.
    ndim_fft = params["ndim_fft_start"]
    N = (
        (params["high_res_n_eff"]) ** (1.0 / 3)
        if params["is_zoom"]
        else (params["n_particles"]) ** (1 / 3.0)
    )
    while float(ndim_fft) / float(N) < params["fft_times_fac"]:
        ndim_fft *= 2
    print("Using ndim_fft = %d" % ndim_fft)
    params["ndim_fft"] = ndim_fft

    # Is this a zoom simulation (zoom can't use 2LPT)?
    if params["is_zoom"]:
        if params["is_slab"]:
            params["two_lpt"] = 1
            params["multigrid"] = 0
        else:
            params["two_lpt"] = 0 if params["multigrid_ics"] else 1
            params["multigrid"] = 1 if params["multigrid_ics"] else 0
    else:
        params["high_res_L"] = 0.0
        params["high_res_n_eff"] = 0
        params["two_lpt"] = 1
        params["multigrid"] = 0

    # Use peano hilbert indexing?
    params["use_ph"] = 2 if params["use_ph_ids"] else 1

    # Cube of neff
    params["high_res_n_eff_cube"] = round(params["high_res_n_eff"] ** (1.0 / 3))

    # Replace template values.
    with open(
        f"./templates/ic_gen/params_{params['num_constraints']}_con.inp", "r"
    ) as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/params.inp" % (ic_gen_dir), "w") as f:
        f.write(result)


# |-------------------------------------------|
# | Make submit and parameter file for SWIFT. |
# |-------------------------------------------|


def make_param_file_swift(params):

    # Make data dir.
    data_dir = params["swift_dir"] + "%s/" % params["f_name"]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(data_dir + "out_files/"):
        os.makedirs(data_dir + "out_files/")
    if not os.path.exists(data_dir + "fof/"):
        os.makedirs(data_dir + "fof/")
    if not os.path.exists(data_dir + "snapshots/"):
        os.makedirs(data_dir + "snapshots/")

    # Starting and finishing scale factors.
    params["starting_a"] = 1.0 / (1 + float(params["starting_z"]))
    params["finishing_a"] = 1.0 / (1 + float(params["finishing_z"]))

    # Replace values.
    if (
        params["template_set"].lower() == "sibelius"
        or params["template_set"].lower() == "sibelius_flamingo"
        or params["template_set"].lower() == "manticore"
    ):
        # Copy over select output.
        subprocess.call(
            "cp ./templates/swift/%s/select_output.yml %s"
            % (params["template_set"], data_dir),
            shell=True,
        )

        # Copy over snapshots times.
        subprocess.call(
            "cp ./templates/swift/%s/stf_times_a.txt %s/snapshot_times.txt"
            % (params["template_set"], data_dir),
            shell=True,
        )
    elif params["template_set"].lower() == "eaglexl":
        raise Exception("Fix this one")
        # split_mass = gas_particle_mass / 10**10. * 4.
        # r = [fname, '%.5f'%h, '%.8f'%starting_a, '%.8f'%finishing_a, '%.8f'%omega0, '%.8f'%omegaL,
        #'%.8f'%omegaB, fname, '%.8f'%(eps_dm/h),
        #'%.8f'%(eps_baryon/h), '%.8f'%(eps_dm_physical/h),
        #'%.8f'%(eps_baryon_physical/h), '%.3f'%(softening_ratio_background),
        #'%.8f'%split_mass, ic_dir, fname]
    else:
        raise ValueError("Invalid template set")

    # Some extra params to compute.
    if params["template_set"].lower() == "sibelius_flamingo":
        params["split_mass"] = params["gas_particle_mass"] / 10**10.0 * 4.0

    t_file = "./templates/swift/%s/params.yml" % params["template_set"]

    # Replace template values.
    with open(t_file, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/params.yml" % (data_dir), "w") as f:
        f.write(result)


def make_submit_file_swift(params):

    data_dir = params["swift_dir"] + "%s/" % params["f_name"]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    s_file = "./templates/swift/%s/submit" % params["template_set"].lower()
    rs_file = "./templates/swift/%s/resubmit" % params["template_set"].lower()

    # Replace template values.
    with open(s_file, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/submit" % (data_dir), "w") as f:
        f.write(result)

    # Replace template values.
    with open(rs_file, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/resubmit" % (data_dir), "w") as f:
        f.write(result)

    with open("%s/auto_resubmit" % data_dir, "w") as f:
        f.write("sbatch resubmit")

    # Change execution privileges (make files executable by group)
    # Assumes the files already exist. If not, it has no effect.
    os.chmod(f"{data_dir}/submit", 0o744)
    os.chmod(f"{data_dir}/resubmit", 0o744)
    os.chmod(f"{data_dir}/auto_resubmit", 0o744)
