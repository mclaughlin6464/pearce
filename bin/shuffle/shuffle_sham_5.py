import numpy as np
import astropy
from itertools import izip
from pearce.mocks import compute_prim_haloprop_bins, cat_dict
from pearce.mocks.customHODModels import *
from halotools.utils.table_utils import compute_conditional_percentiles
from halotools.utils import *

PMASS = 591421440.0000001 #chinchilla 400/ 2048
Lbox = 400.0
catalog = astropy.table.Table.read('/u/ki/swmclau2/des/AB_tests/abmatched_halos.hdf5', format = 'hdf5')
catalog = catalog[catalog['halo_mvir'] > 200*PMASS]

add_halo_hostid(catalog, delete_possibly_existing_column=True)

for prop in ['halo_x', 'halo_y', 'halo_z', 'halo_nfw_conc', 'halo_mvir', 'halo_rvir']:
    broadcast_host_halo_property(catalog, prop, delete_possibly_existing_column=True)

from halotools.utils.table_utils import compute_prim_haloprop_bins
from math import ceil
min_log_mass = np.log10(np.min(catalog['halo_mvir']))-0.001
max_log_mass = np.log10(np.max(catalog['halo_mvir']))+0.001
dlog10_prim_haloprop = 0.1
num_prim_haloprop_bins = (max_log_mass - min_log_mass) / dlog10_prim_haloprop
prim_haloprop_bin_boundaries = np.logspace(min_log_mass, max_log_mass,
    num=int(ceil(num_prim_haloprop_bins)))

prim_haloprop_bins = compute_prim_haloprop_bins(prim_haloprop = catalog['halo_mvir_host_halo'],                                                dlog10_prim_haloprop=dlog10_prim_haloprop,
                                                prim_haloprop_bin_boundaries = prim_haloprop_bin_boundaries)


shuffled_pos = np.zeros((len(catalog), 3))
shuffled_upids = np.zeros((len(catalog)))
shuffled_host_mvir = np.zeros((len(catalog)))
shuffled_mags = np.zeros((len(catalog), 3))

from pearce.mocks import cat_dict
cosmo_params = {'simname':'chinchilla', 'Lbox':400.0, 'scale_factors':[0.658, 1.0]}
cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!
cat.load_model(1.0, HOD = 'redMagic')

np.random.seed(64)
bins_in_halocat = set(prim_haloprop_bins)

for ibin in bins_in_halocat:
    print ibin
    if ibin==0:
        continue
    indices_of_prim_haloprop_bin = np.where(prim_haloprop_bins == ibin)[0]
    
    centrals_idx = np.where(catalog[indices_of_prim_haloprop_bin]['halo_upid'] == -1)[0]
    n_centrals = len(centrals_idx)
    satellites_idx = np.where(catalog[indices_of_prim_haloprop_bin]['halo_upid']!=-1)[0]
    n_satellites = len(satellites_idx)
    
    if centrals_idx.shape[0]!=0:
        rand_central_idxs = np.random.choice(indices_of_prim_haloprop_bin[centrals_idx], size = n_centrals, replace = False)
    else:
        rand_central_idxs = np.array([])

    for idx, coord in enumerate(['vpeak', 'vvir', 'alpha_05']):
        shuffled_mags[indices_of_prim_haloprop_bin[centrals_idx], idx]=                 catalog[rand_central_idxs]['halo_'+coord+'_mag']
            
        shuffled_mags[indices_of_prim_haloprop_bin[satellites_idx],idx ] =                 catalog[indices_of_prim_haloprop_bin[satellites_idx]]['halo_'+coord+'_mag']
    #Create second rand_central_idxs, Iterate through satellite hosts and assign them when they match. 
    
    for idx, coord in enumerate(['x','y','z']):
        # don't need to shuffle positions cu we've shuffled mags for centrals
        shuffled_pos[indices_of_prim_haloprop_bin[centrals_idx], idx] =                 catalog[indices_of_prim_haloprop_bin[centrals_idx]]['halo_'+coord]
            
    shuffled_upids[indices_of_prim_haloprop_bin[centrals_idx]] = -1
    
    shuffled_host_mvir[indices_of_prim_haloprop_bin[centrals_idx]] =             catalog[indices_of_prim_haloprop_bin[centrals_idx]]['halo_mvir']
        
    hosts_id, first_sat_idxs = np.unique(catalog[indices_of_prim_haloprop_bin[satellites_idx]]['halo_upid'], return_index=True)
    shuffled_idxs = np.random.permutation(hosts_id.shape[0])
    shuffled_hosts_id = hosts_id[shuffled_idxs]
    shuffled_sat_idxs = first_sat_idxs[shuffled_idxs]
    shuffled_arrays_idx = 0
    host_map = dict() #maps the current host id to the index of a new host id. 
    #the host_id -> idx map is easier than the host_id -> host_id map

    for sat_idx in satellites_idx:
        host_id = catalog[indices_of_prim_haloprop_bin][sat_idx]['halo_upid']
        
        if host_id in host_map:
            new_host_id, hosts_old_satellite_idx = host_map[host_id]
        else:
            new_host_id = shuffled_hosts_id[shuffled_arrays_idx]
            hosts_old_satellite_idx = shuffled_sat_idxs[shuffled_arrays_idx]
            host_map[host_id] = (new_host_id, hosts_old_satellite_idx)
            shuffled_arrays_idx+=1
            
        shuffled_upids[indices_of_prim_haloprop_bin[sat_idx]] = new_host_id
        
        shuffled_host_mvir[indices_of_prim_haloprop_bin[sat_idx]] =                 catalog[indices_of_prim_haloprop_bin[satellites_idx]][hosts_old_satellite_idx]['halo_mvir_host_halo']


        hc_x, hc_y, hc_z = cat.model.model_dictionary['satellites_profile'].mc_halo_centric_pos(                                                        catalog[indices_of_prim_haloprop_bin[sat_idx]]['halo_nfw_conc_host_halo'],
                                                        halo_radius = catalog[indices_of_prim_haloprop_bin[sat_idx]]['halo_rvir_host_halo'])
        
        
        for idx, (coord, hc) in enumerate(izip(['x','y','z'], [hc_x, hc_y, hc_z])):
            #shuffled_pos[indices_of_prim_haloprop_bin[satellites_idx], idx] = \
            #        (catalog[indices_of_prim_haloprop_bin[satellites_idx]]['halo_'+coord] -\
            #        host_halo_pos[indices_of_prim_haloprop_bin[satellites_idx], idx]+\
            #        host_halo_pos[rand_host_idxs, idx])%Lbox

            shuffled_pos[indices_of_prim_haloprop_bin[sat_idx],idx] =                        (catalog[indices_of_prim_haloprop_bin[satellites_idx][hosts_old_satellite_idx]]['halo_'+coord+'_host_halo'] + hc)%Lbox

catalog['halo_sh_shuffled_vpeak_mag'] = shuffled_mags[:,0]
catalog['halo_sh_shuffled_vvir_mag'] = shuffled_mags[:,1]
catalog['halo_sh_shuffled_alpha_05_mag'] = shuffled_mags[:,2]
catalog['halo_sh_shuffled_x'] = shuffled_pos[:,0]
catalog['halo_sh_shuffled_y'] = shuffled_pos[:,1]
catalog['halo_sh_shuffled_z'] = shuffled_pos[:,2]
catalog['halo_sh_shuffled_upid']=shuffled_upids[:]
catalog['halo_sh_shuffled_host_mvir'] = shuffled_host_mvir[:]

catalog.write('/u/ki/swmclau2/des/AB_tests/abmatched_halos.hdf5', format = 'hdf5', path = './abmatched_halos.hdf5', overwrite=True)
