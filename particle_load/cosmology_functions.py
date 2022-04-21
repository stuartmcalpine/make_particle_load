from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


def compute_masses(pl_params):
    """For the given cosmology, compute the total DM mass for the given volume."""

    h = pl_params.HubbleParam
    Om0 = pl_params.Omega0
    Ob0 = pl_params.OmegaBaryon
    box_size = pl_params.box_size

    cosmo = FlatLambdaCDM(H0=h * 100.0, Om0=Om0, Ob0=Ob0)
    rho_crit = cosmo.critical_density0.to(u.solMass / u.Mpc**3)

    M_tot_dm_dmo = Om0 * rho_crit.value * (box_size / h) ** 3
    M_tot_dm = (Om0 - Ob0) * rho_crit.value * (box_size / h) ** 3
    M_tot_gas = Ob0 * rho_crit.value * (box_size / h) ** 3

    dm_mass = M_tot_dm / pl_params.n_particles
    dm_mass_dmo = M_tot_dm_dmo / pl_params.n_particles
    gas_mass = M_tot_gas / pl_params.n_particles

    print(
        "Dark matter particle mass (if DMO): %.3g Msol (%.3g 1e10 Msol/h)"
        % (dm_mass_dmo, dm_mass_dmo * h / 1.0e10)
    )
    print(
        "Dark matter particle mass: %.3g Msol (%.3g 1e10 Msol/h)"
        % (dm_mass, dm_mass * h / 1.0e10)
    )
    print(
        "Gas particle mass: %.3g Msol (%.3g 1e10 Msol/h)"
        % (gas_mass, gas_mass * h / 1.0e10)
    )

    pl_params.total_box_mass = M_tot_dm_dmo
    pl_params.gas_particle_mass = gas_mass


def compute_softening(pl_params):
    """Compute softning legnths."""

    if "flamingo" in pl_params.template_set.lower():
        # Flamingo.
        comoving_ratio = 1 / 25.0
        physical_ratio = 1 / 100.0  # Transition at z=3.
    else:
        # EagleXL.
        comoving_ratio = 1 / 20.0
        physical_ratio = 1 / 45.0  # Transition at z = 1.25

    N = pl_params.n_particles ** (1 / 3.0)
    mean_inter = pl_params.box_size / N

    # DM
    pl_params.eps_dm = mean_inter * comoving_ratio
    pl_params.eps_dm_physical = mean_inter * physical_ratio

    # Baryons
    if "flamingo" in pl_params.template_set.lower():
        pl_params.eps_baryon = pl_paramseps_dm
        pl_params.eps_baryon_physical = pl_params.eps_dm_physical
    else:
        fac = ((pl_params.Omega0 - pl_params.OmegaBaryon) / pl_params.OmegaBaryon) ** (
            1.0 / 3
        )
        pl_params.eps_baryon = pl_params.eps_dm / fac
        pl_params.eps_baryon_physical = pl_params.eps_dm_physical / fac

    print(
        "Comoving Softenings: DM=%.6f Baryons=%.6f Mpc/h"
        % (pl_params.eps_dm, pl_params.eps_baryon)
    )
    print(
        "Max phys Softenings: DM=%.6f Baryons=%.6f Mpc/h"
        % (pl_params.eps_dm_physical, pl_params.eps_baryon_physical)
    )
    print(
        "Comoving Softenings: DM=%.6f Baryons=%.6f Mpc"
        % (
            pl_params.eps_dm / pl_params.HubbleParam,
            pl_params.eps_baryon / pl_params.HubbleParam,
        )
    )
    print(
        "Max phys Softenings: DM=%.6f Baryons=%.6f Mpc"
        % (
            pl_params.eps_dm_physical / pl_params.HubbleParam,
            pl_params.eps_baryon_physical / pl_params.HubbleParam,
        )
    )
