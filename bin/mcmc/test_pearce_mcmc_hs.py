from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks.customHODModels import *
from pearce.mocks import cat_dict
from pearce.inference import run_mcmc
from halotools.mock_observables import wp
from astropy.table import Table
import numpy as np
from os import path

training_dir = '/u/ki/swmclau2/des/PearceLHC_wp_z_sham_tabulated_no_split/'

em_method = 'gp'
split_method = 'random'

load_fixed_params = {'z':0.0}

emu = ExtraCrispy(training_dir,10, 2, split_method, method=em_method, fixed_params=load_fixed_params)

#Remember if training data is an LHC can't load a fixed set, do that after
fixed_params = {}#{'f_c':1.0}#,'logM1': 13.8 }# 'z':0.0}

cosmo_params = {'simname':'chinchilla', 'Lbox':400.0, 'scale_factors':[1.0]}
cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!
mbc = np.loadtxt('/nfs/slac/g/ki/ki18/des/swmclau2/AB_tests/mbc.npy')
cen_hod = np.loadtxt('/nfs/slac/g/ki/ki18/des/swmclau2/AB_tests/cen_hod.npy')
sat_hod = np.loadtxt('/nfs/slac/g/ki/ki18/des/swmclau2/AB_tests/sat_hod.npy')

cat.load_model(1.0, HOD=(FSAssembiasTabulatedCens, FSAssembiasTabulatedSats),\
                hod_kwargs = {'prim_haloprop_vals': mbc,
                              'cen_hod_vals':cen_hod,
                              'sat_hod_vals':sat_hod})
cat.load_catalog(1.0)
#cat.load(1.0, HOD='redMagic')
#emulation_point = [('f_c', 0.2), ('logM0', 12.0), ('sigma_logM', 0.366), 
#                    ('alpha', 1.083),('logM1', 13.7), ('logMmin', 12.233)]
emulation_point = [('mean_occupation_centrals_assembias_param1',0.6),\
                    ('mean_occupation_satellites_assembias_param1',-0.7)]

em_params = dict(emulation_point)


em_params.update(fixed_params)
#del em_params['z']

#rp_bins =  np.logspace(-1.1,1.6,18) 
#rp_bins.pop(1)
#rp_bins = np.array(rp_bins)
#rp_bins = np.loadtxt('/nfs/slac/g/ki/ki18/des/swmclau2/AB_tests/rp_bins_split.npy')
rp_bins = np.loadtxt(training_dir+'a_1.00000/global_file.npy')
rpoints = (rp_bins[1:]+rp_bins[:-1])/2.0
#compute the sham clustering and nd here unambiguously
shuffle_type = ''
mag_type = 'vpeak'
mag_key = 'halo%s_%s_mag'%(shuffle_type, mag_type)

PMASS = 591421440.0000001 #chinchilla 400/ 2048
halo_catalog = Table.read('/u/ki/swmclau2/des/AB_tests/abmatched_halos.hdf5', format = 'hdf5')

mag_cut = -21
min_ptcl = 200

halo_catalog = halo_catalog[halo_catalog['halo_mvir'] > min_ptcl*cat.pmass] #mass cut
galaxy_catalog = halo_catalog[ halo_catalog[mag_key] < mag_cut ] # mag cut

sham_pos = np.c_[galaxy_catalog['halo%s_x'%shuffle_type],\
                 galaxy_catalog['halo%s_y'%shuffle_type],\
                 galaxy_catalog['halo%s_z'%shuffle_type]]

y = np.log10(wp(sham_pos*cat.h, rp_bins, 40.0*cat.h, period=cat.Lbox*cat.h, num_threads=1))
obs_nd = len(galaxy_catalog)*1.0/((cat.Lbox*cat.h)**3)

wp_vals = []
nds = []
for i in xrange(10):
    cat.populate(em_params)
    wp_vals.append(cat.calc_wp(rp_bins, 40))
    nds.append(cat.calc_number_density())
#y = np.mean(np.log10(np.array(wp_vals)),axis = 0 )
#y = np.log10(np.loadtxt('/nfs/slac/g/ki/ki18/des/swmclau2/AB_tests/sham_vpeak_wp.npy'))
# TODO need a way to get a measurement cov for the shams
cov = np.cov(np.log10(np.array(wp_vals).T))#/np.sqrt(50)

#obs_nd = np.mean(np.array(nds))
#obs_nd = np.loadtxt('/nfs/slac/g/ki/ki18/des/swmclau2/AB_tests/sham_vpeak_nd.npy')
obs_nd_err = np.std(np.array(nds))

param_names = [k for k in em_params.iterkeys() if k not in fixed_params]

nwalkers = 100
nsteps = 1000
nburn = 0 

chain = run_mcmc(emu, cat, param_names, y, cov, rpoints,obs_nd, obs_nd_err,'calc_analytic_nd', fixed_params = fixed_params,\
        nwalkers = nwalkers, nsteps = nsteps, nburn = nburn)#,\
        #resume_from_previous = '/u/ki/swmclau2/des/PearceMCMC/100_walkers_1000_steps_chain_vpeak_sham_free_split_2.npy')#, ncores = 1)
#from sys import exit
#exit(1)

savedir = '/u/ki/swmclau2/des/PearceMCMC/'
np.savetxt(path.join(savedir, '%d_walkers_%d_steps_chain_vpeak_sham_hs.npy'%(nwalkers, nsteps)), chain)
#np.savetxt(path.join(savedir, '%d_walkers_%d_steps_truth_ld_errors_2.npy'%(nwalkers, nsteps)),\
#                                np.array([em_params[p] for p in param_names]))
#np.savetxt(path.join(savedir, '%d_walkers_%d_steps_fixed_old_errors_2.npy'%(nwalkers, nsteps)),\
#                                np.array([fixed_params[p] for p in param_names if p in fixed_params]))


