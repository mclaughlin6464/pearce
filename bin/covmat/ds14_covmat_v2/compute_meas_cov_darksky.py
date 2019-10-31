from pearce.mocks.kittens import DarkSky 
from halotools.empirical_models import Zheng07Cens, Zheng07Sats
from halotools.mock_observables.surface_density.surface_density_helpers import rho_matter_comoving_in_halotools_units as rho_m_comoving
from halotools.mock_observables.surface_density.surface_density_helpers import annular_area_weighted_midpoints
from halotools.mock_observables.surface_density.surface_density_helpers import log_interpolation_with_inner_zero_masking as log_interp
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables.surface_density.delta_sigma import _delta_sigma_precomputed_process_args
from halotools.mock_observables.surface_density.mass_in_cylinders import _enclosed_mass_process_args
from halotools.mock_observables.surface_density.weighted_npairs_per_object_xy import weighted_npairs_per_object_xy
from itertools import product
import numpy as np
from time import time
from scipy.optimize import minimize_scalar
import yaml
import sys

def make_LHC(ordered_params, N, seed = None):

    if seed is None:
        seed = int(time())
    np.random.seed(seed)

    points = []
    # by linspacing each parameter and shuffling, I ensure there is only one point in each row, in each dimension.
    for plow, phigh in ordered_params.itervalues():
        point = np.linspace(plow, phigh, num=N)
        np.random.shuffle(point)  # makes the cube random.
        points.append(point)
    return np.stack(points).T

def add_logMmin(hod_params, cat):

    hod_params['logMmin'] = 13.0 #initial guess
    #cat.populate(hod_params) #may be overkill, but will ensure params are written everywhere
    def func(logMmin, hod_params):
        hod_params.update({'logMmin':logMmin}) 
        return (cat.calc_analytic_nd(hod_params, min_ptcl = min_ptcl) - nd)**2

    res = minimize_scalar(func, bounds = logMmin_bounds, args = (hod_params,), options = {'maxiter':100}, method = 'Bounded')

    # assuming this doens't fail
    #print 'logMmin', res.x
    hod_params['logMmin'] = res.x

def total_mass_enclosed_per_cylinder(centers, particles,
        particle_masses, downsampling_factor, rp_bins, period,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):

#  Perform bounds-checking and error-handling in private helper functions
    #print period
    args = (centers, particles, particle_masses, downsampling_factor,
        rp_bins, period, num_threads)
    result = _enclosed_mass_process_args(*args)
    centers, particles, particle_masses, downsampling_factor, \
        rp_bins, period, num_threads, PBCs = result


    mean_particle_mass = np.mean(particle_masses)
    normalized_particle_masses = particle_masses/mean_particle_mass

    # Calculate M_tot(< Rp) normalized with internal code units
    total_mass_per_cylinder = weighted_npairs_per_object_xy(centers, particles,
        normalized_particle_masses, rp_bins,
        period=None, num_threads=num_threads, #try large finite PBCs
        approx_cell1_size=approx_cell1_size,
        approx_cell2_size=approx_cell2_size)

    # Renormalize the particle masses and account for downsampling
    total_mass_per_cylinder *= downsampling_factor*mean_particle_mass

    return total_mass_per_cylinder

def calc_ds(cat, rp_bins, n_cores, tm_rand=None, randoms = None):

    x_g, y_g, z_g = [cat.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]
    pos_g = return_xyz_formatted_array(x_g, y_g, z_g, period=cat.Lbox)
    x_m, y_m, z_m = [cat.halocat.ptcl_table[c] for c in ['x', 'y', 'z']]
    pos_m = return_xyz_formatted_array(x_m, y_m, z_m, period=cat.Lbox)

    tm_gal = total_mass_enclosed_per_cylinder(pos_g / cat.h, pos_m / cat.h,
                                                  cat.pmass / cat.h, 1. / cat._downsample_factor, rp_bins,
                                                  cat.Lbox / cat.h,
                                                  num_threads=n_cores)
    if tm_rand is None:
        assert randoms is not None
        tm_rand = total_mass_enclosed_per_cylinder(randoms / cat.h, pos_m / cat.h,
                                                   cat.pmass / cat.h, 1. / cat._downsample_factor, rp_bins,
                                                   cat.Lbox / cat.h,
                                                   num_threads=n_cores)
    if tm_rand.shape[0] > tm_gal.shape[0]:
        tm_rand_idxs = np.random.choice(tm_rand.shape[0], tm_gal.shape[0], replace = False)
        tm_rand = tm_rand[tm_rand_idxs]

    elif tm_rand.shape[0] < tm_gal.shape[0]:
        raise AssertionError("Need more randoms!")

    return delta_sigma(pos_g / cat.h, tm_gal, tm_rand,
                       cat.Lbox / cat.h, rp_bins, cosmology=cat.cosmology) / (1e12)#, tm_gal, tm_rand


def delta_sigma(galaxies, mass_enclosed_per_galaxy,
                mass_enclosed_per_random, period,
                rp_bins, cosmology):
    #  Perform bounds-checking and error-handling in private helper functions
    args = (galaxies, mass_enclosed_per_galaxy, rp_bins, period)
    result = _delta_sigma_precomputed_process_args(*args)
    galaxies, mass_enclosed_per_galaxy, rp_bins, period, PBCs = result

    total_mass_in_stack_of_cylinders = np.sum(mass_enclosed_per_galaxy, axis=0)

    total_mass_in_stack_of_annuli = np.diff(total_mass_in_stack_of_cylinders)

    mean_rho_comoving = rho_m_comoving(cosmology)
    mean_sigma_comoving = mean_rho_comoving * float(period[2])

    expected_mass_in_random_stack_of_cylinders = np.sum(mass_enclosed_per_random, axis=0)
    expected_mass_in_random_stack_of_annuli = np.diff(expected_mass_in_random_stack_of_cylinders)

    one_plus_mean_sigma_inside_rp = mean_sigma_comoving * (
            total_mass_in_stack_of_cylinders / expected_mass_in_random_stack_of_cylinders)

    one_plus_sigma = mean_sigma_comoving * (
            total_mass_in_stack_of_annuli / expected_mass_in_random_stack_of_annuli)

    rp_mids = annular_area_weighted_midpoints(rp_bins)
    one_plus_mean_sigma_inside_rp_interp = log_interp(one_plus_mean_sigma_inside_rp,
                                                      rp_bins, rp_mids)

    excess_surface_density = one_plus_mean_sigma_inside_rp_interp - one_plus_sigma
    return excess_surface_density


def compute_obs(cat, rp_bins, cic_bins, randoms, total_mass_randoms=None):

    n_cores = cat._check_cores(1)

    wp = cat.calc_wp(rp_bins, PBC=False, randoms=randoms/cat.h, n_cores = n_cores)
    # have to make a custom function to do no PBC
    ds = calc_ds(cat, rp_bins, n_cores=n_cores, randoms = randoms)# tm_rand = total_mass_randoms)
    cic = cat.calc_cic(cic_bins, PBC=False, n_cores = n_cores)
    #print len(ds), len(cic)
    return np.r_[wp, ds, cic]

config_fname = 'xi_cosmo_trainer.yaml'

with open(config_fname, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

nd = float(cfg['HOD']['fixed_nd'] )
min_ptcl = int(cfg['HOD']['min_ptcl'])

rp_bins = np.logspace(-1.0, 1.6, 19)
cic_bins = np.round(np.r_[np.linspace(1, 9, 8), np.round(np.logspace(1,2, 7))])

hod_param_ranges =  cfg['HOD']['ordered_params'] 

N = 5 
LHC = make_LHC(hod_param_ranges, N, 16)# 23)
hod_dicts = [dict(zip(hod_param_ranges.keys(), vals)) for vals in LHC]

logMmin_bounds = hod_param_ranges['logMmin']

del hod_param_ranges['logMmin']

obs_vals = np.zeros((N, 2*(len(rp_bins)-1)+len(cic_bins)-1))

HOD = (Zheng07Cens, Zheng07Sats)

np.random.seed(23)
randoms = np.random.rand(int(5e6), 3)

#total_mass_randoms = np.load('total_mass_randoms.npy')

b1, b2, b3 = sys.argv[1], sys.argv[2], sys.argv[3]
start_subbox = (b1,  b2, b3)

print 'A'

start_idx = 64*int(start_subbox[0])+8*int(start_subbox[1])+int(start_subbox[2])
for subbox_idx, subbox in enumerate(product(''.join([str(i) for i in xrange(8)]), repeat = 3)):
    if subbox_idx < start_idx:
        continue
    print 'B'
    cat = DarkSky(int(''.join(subbox)), system = 'sherlock')
    setattr(cat, '_downsample_factor', 1e-2)
    print 'C'
    cat.load_model(1.0, HOD = HOD, hod_kwargs = {'modlulate_with_cenocc': True})
    print 'D'
    cat.load_catalog_no_cache(1.0, min_ptcl=min_ptcl, particles = True)#, downsample_factor = 1e-2)
    print 'E'
    for hod_idx, hod_params in enumerate(hod_dicts):
        print 'HOD: %d'%hod_idx
        add_logMmin(hod_params, cat)
        cat.populate(hod_params, min_ptcl = min_ptcl)
        sys.stdout.flush()
        obs = compute_obs(cat, rp_bins, cic_bins,  randoms*cat.Lbox)#, total_mass_randoms)
        obs_vals[hod_idx] = obs

        np.save('wp_ds_cic_darksky_obs_%s%s%s_v2.npy'%(b1,b2,b3), obs_vals)

    break
