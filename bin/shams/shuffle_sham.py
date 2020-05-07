
# coding: utf-8

# Change the positions of the galaxies in a SHAM to be shuffled then NFW-distributed, instead of on the subhalos. 
# 
# Shuffling procedure is as followed, from Jeremey
# 
# procedure:
# 
# take a bin in halo mass (small bins, like 0.1dex wide). (this is all halos, regardless of whether they have a galaxy in them or not). take all the centrals and put them in a list. take all the satellites and put them in a separate list.
# 
# randomly assign the centrals to all the halos in the bin.
# 
# randomly assign each satellite to a halo in the bin (repeat until all satellites are gone. this should preserve poisson distribution of satellite occupation). when assigning a satellite to a halo, preserve the position of the satellite and velocity of the satellite relative to the original host halo. ie, your list of satllites has dx, dy, dz, and dvx, dvy, dvz, then you add x, y, z, and vx, vy, vz of the new halo to those quantities.

# In[1]:


import numpy as np
import astropy
from itertools import izip
from halotools.utils.table_utils import compute_conditional_percentiles
from halotools.utils import *


Lbox = 1000.0

#ab_property = 'halo_mpeak'
#ab_property = 'halo_vmax@mpeak'
#catalog = astropy.table.Table.read('/scratch/users/swmclau2/catalog_ab_%s_large_fixed.hdf5'%ab_property, format = 'hdf5')
ab_property = 'halo_vpeak'
catalog = astropy.table.Table.read('/scratch/users/swmclau2/test_MDPL2_%s_smf_sham_large.hdf5'%ab_property, format = 'hdf5')

#PMASS = 7.62293e+07
nd = 5e-4#4.2e-4 #nd of final cat 
n_obj_needed = int(nd*(Lbox**3))


add_halo_hostid(catalog, delete_possibly_existing_column=True)


for prop in ['halo_x', 'halo_y', 'halo_z','halo_vx', 'halo_vy', 'halo_vz', 'halo_nfw_conc', 'halo_rvir']:
    try:
        broadcast_host_halo_property(catalog, prop, delete_possibly_existing_column=True)
    except:
        pass

# lifted from halotools
def compute_prim_haloprop_bins(dlog10_prim_haloprop=0.05, **kwargs):
    r"""
    Parameters
    ----------
    prim_haloprop : array
        Array storing the value of the primary halo property column of the ``table``
        passed to ``compute_conditional_percentiles``.
    prim_haloprop_bin_boundaries : array, optional
        Array defining the boundaries by which we will bin the input ``table``.
        Default is None, in which case the binning will be automatically determined using
        the ``dlog10_prim_haloprop`` keyword.
    dlog10_prim_haloprop : float, optional
        Logarithmic spacing of bins of the mass-like variable within which
        we will assign secondary property percentiles. Default is 0.2.
    Returns
    --------
    output : array
        Numpy array of integers storing the bin index of the prim_haloprop bin
        to which each halo in the input table was assigned.
    """
    try:
        prim_haloprop = kwargs['prim_haloprop']
    except KeyError:
        msg = ("The ``compute_prim_haloprop_bins`` method "
            "requires the ``prim_haloprop`` keyword argument")
        raise HalotoolsError(msg)

    try:
        prim_haloprop_bin_boundaries = kwargs['prim_haloprop_bin_boundaries']
    except KeyError:
        lg10_min_prim_haloprop = np.log10(np.min(prim_haloprop))-0.001
        lg10_max_prim_haloprop = np.log10(np.max(prim_haloprop))+0.001
        num_prim_haloprop_bins = (lg10_max_prim_haloprop-lg10_min_prim_haloprop)/dlog10_prim_haloprop
        prim_haloprop_bin_boundaries = np.logspace(
            lg10_min_prim_haloprop, lg10_max_prim_haloprop,
            num=int(ceil(num_prim_haloprop_bins)))

    # digitize the masses so that we can access them bin-wise
    output = np.digitize(prim_haloprop, prim_haloprop_bin_boundaries)

    # Use the largest bin for any points larger than the largest bin boundary,
    # and raise a warning if such points are found
    Nbins = len(prim_haloprop_bin_boundaries)
    if Nbins in output:
        msg = ("\n\nThe ``compute_prim_haloprop_bins`` function detected points in the \n"
            "input array of primary halo property that were larger than the largest value\n"
            "of the input ``prim_haloprop_bin_boundaries``. All such points will be assigned\n"
            "to the largest bin.\nBe sure that this is the behavior you expect for your application.\n\n")
        warn(msg)
        output = np.where(output == Nbins, Nbins-1, output)

    return output




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
shuffled_vel = np.zeros((len(catalog), 3))
shuffled_ids = np.zeros((len(catalog)))
shuffled_upids = np.zeros((len(catalog)))
shuffled_host_mvir = np.zeros((len(catalog)))


# In[ ]:


shuffled_mags = np.zeros((len(catalog), 1))
#shuffled_mags[:, 0] = catalog['halo_vpeak_mag']
#shuffled_mags[:, 1] = catalog['halo_vvir_mag']
#shuffled_mags[:, 2] = catalog['halo_alpha_05_mag']


# In[ ]:


bins_in_halocat = set(prim_haloprop_bins)

for ibin in bins_in_halocat:
    print ibin
    #if ibin > 25:
    #    continue
    indices_of_prim_haloprop_bin = np.where(prim_haloprop_bins == ibin)[0]
    
    centrals_idx = np.where(catalog[indices_of_prim_haloprop_bin]['halo_upid'] == -1)[0]
    n_centrals = len(centrals_idx)
    satellites_idx = np.where(catalog[indices_of_prim_haloprop_bin]['halo_upid']!=-1)[0]
    n_satellites = len(satellites_idx)
    
    if centrals_idx.shape[0]!=0:
        rand_central_idxs = np.random.choice(indices_of_prim_haloprop_bin[centrals_idx], size = n_centrals, replace = False)
    else:
        rand_central_idxs = np.array([])
        
    shuffled_mags[indices_of_prim_haloprop_bin[centrals_idx],0]=             catalog[rand_central_idxs]['gal_smass']

    shuffled_mags[indices_of_prim_haloprop_bin[satellites_idx],0] =             catalog[indices_of_prim_haloprop_bin[satellites_idx]]['gal_smass']
    #Create second rand_central_idxs, Iterate through satellite hosts and assign them when they match. 
                
    for idx, coord in enumerate(['x','y','z']):
        # don't need to shuffle positions cu we've shuffled mags for centrals
        shuffled_pos[indices_of_prim_haloprop_bin[centrals_idx], idx] =                 catalog[indices_of_prim_haloprop_bin[centrals_idx]]['halo_'+coord]
            
        shuffled_vel[indices_of_prim_haloprop_bin[centrals_idx], idx] =                 catalog[indices_of_prim_haloprop_bin[centrals_idx]]['halo_v'+coord]
            
    shuffled_ids[indices_of_prim_haloprop_bin[centrals_idx]] =  catalog[indices_of_prim_haloprop_bin[centrals_idx]]['halo_id']

    shuffled_upids[indices_of_prim_haloprop_bin[centrals_idx]] = -1
    
    shuffled_host_mvir[indices_of_prim_haloprop_bin[centrals_idx]] =             catalog[indices_of_prim_haloprop_bin[centrals_idx]]['halo_mvir']
        
    unique_hosts_id, first_sat_idxs, inverse_idxs = np.unique(catalog[indices_of_prim_haloprop_bin[satellites_idx]]['halo_upid'],                                                       return_index=True, return_inverse=True)

    shuffled_idxs = np.random.permutation(unique_hosts_id.shape[0])
    shuffled_hosts_id = unique_hosts_id[shuffled_idxs]
    shuffled_sat_idxs = first_sat_idxs[shuffled_idxs]
    shuffled_arrays_idx = 0
    host_map = dict() #maps the current host id to the index of a new host id. 
    #the host_id -> idx map is easier than the host_id -> host_id map
    
    new_host_ids = shuffled_hosts_id[inverse_idxs]
    hosts_old_satellite_idxs = shuffled_sat_idxs[inverse_idxs]
            
    shuffled_ids[indices_of_prim_haloprop_bin[satellites_idx]] = -1 # TODO not -1, but also not important i think  
    shuffled_upids[indices_of_prim_haloprop_bin[satellites_idx]] = new_host_ids

    shuffled_host_mvir[indices_of_prim_haloprop_bin[satellites_idx]] =             catalog[indices_of_prim_haloprop_bin[satellites_idx]][hosts_old_satellite_idxs]['halo_mvir_host_halo']

    for idx, coord in enumerate(['x','y','z']):

        shuffled_pos[indices_of_prim_haloprop_bin[satellites_idx], idx] =                 (catalog[indices_of_prim_haloprop_bin[satellites_idx]]['halo_'+coord] -                catalog[indices_of_prim_haloprop_bin[satellites_idx]]['halo_'+coord+'_host_halo']+                catalog[indices_of_prim_haloprop_bin[satellites_idx]][hosts_old_satellite_idxs]['halo_'+coord+'_host_halo'])%Lbox
       
        #print catalog[indices_of_prim_haloprop_bin[sat_idx]]['halo_'+coord]
        #print catalog[indices_of_prim_haloprop_bin[sat_idx]]['halo_'+coord+'_host_halo']
        #print catalog[indices_of_prim_haloprop_bin[satellites_idx]][hosts_old_satellite_idx]['halo_'+coord+'_host_halo']
        #print '*'*50       
        shuffled_vel[indices_of_prim_haloprop_bin[satellites_idx], idx] =                 (catalog[indices_of_prim_haloprop_bin[satellites_idx]]['halo_v'+coord] -                catalog[indices_of_prim_haloprop_bin[satellites_idx]]['halo_v'+coord+'_host_halo']+                catalog[indices_of_prim_haloprop_bin[satellites_idx]][hosts_old_satellite_idxs]['halo_v'+coord+'_host_halo'])

# In[ ]:


catalog['gal_smass'] = shuffled_mags[:,0]
catalog['halo_x'] = shuffled_pos[:,0]
catalog['halo_y'] = shuffled_pos[:,1]
catalog['halo_z'] = shuffled_pos[:,2]
catalog['halo_vx'] = shuffled_vel[:,0]
catalog['halo_vy'] = shuffled_vel[:,1]
catalog['halo_vz'] = shuffled_vel[:,2]
catalog['halo_id'] = shuffled_ids[:]
catalog['halo_upid']=shuffled_upids[:]
catalog['halo_mvir_host_halo'] = shuffled_host_mvir[:]


# In[ ]:
delete_keys = ['halo_vmax@mpeak', 'halo_rvir', 'halo_mpeak', 'halo_id', 'halo_rs', 'halo_nfw_conc', 'halo_hostid',
               'halo_mvir_host_halo', 'halo_x_host_halo', 'halo_y_host_halo', 'halo_z_host_halo', 'halo_vx_host_halo',
               'halo_vy_host_halo', 'halo_vz_host_halo', 'halo_rvir_host_halo', 'halo_nfw_conc_host_halo']
for key in delete_keys:
    try:
        del catalog[key]
    except KeyError:
        continue


sort_idxs = np.argsort(catalog[~np.isnan(catalog['gal_smass'])]['gal_smass'])[::-1]
#catalog = catalog[~np.isnan(catalog['gal_smass'])][sort_idxs[-1*n_obj_needed:]]
catalog = catalog[~np.isnan(catalog['gal_smass'])][sort_idxs[:n_obj_needed]]

# In[ ]:


#plt.hist(catalog[catalog['halo_upid']==-1]['halo_x'], bins = 100);


# In[ ]:


#plt.plot(sorted(catalog['halo_x'][catalog['halo_upid']!=-1]))
#plt.yscale('log')
#plt.ylim([-1, 1001])


# In[ ]:


#from collections import Counter
#c = Counter(catalog['halo_x'][catalog['halo_upid']!=-1])
#print c.most_common(50)


# In[ ]:


from halotools.mock_observables import tpcf


# In[ ]:


rbins = np.logspace(-1, 1.5, 15)
pos = np.c_[catalog['halo_x'], catalog['halo_y'],catalog['halo_z']]
xi = tpcf(pos, rbins, period=1000.0)


# In[ ]:


#simple sanity check
print xi


# In[ ]:


#rbc = (rbins[1:]+rbins[:-1])/2.0
#plt.plot(rbc, xi)

#plt.loglog();
#plt.legend(loc='best')
#plt.xlabel('r [Mpc]')
#plt.ylabel('xi')


# In[ ]:

#catalog.write('/scratch/users/swmclau2/MDPL2_sham_%s_shuffled.hdf5'%ab_property,
#              format = 'hdf5', path = '%s_shuffled'%ab_property, overwrite=True)

np.save('/scratch/users/swmclau2/UniverseMachine/cut_shuffled_sham_catalog.npy', catalog.as_array())

# In[ ]:
