
# coding: utf-8

# ZhongXu has asked that I make him the HODs from the SHAMs I made him. This should be pretty straightforward so I'll do it quickly here.  

# In[1]:


import numpy as np
import astropy
from pearce.mocks import cat_dict
from pearce.mocks.assembias_models.table_utils import compute_prim_haloprop_bins
import h5py


# In[2]:
from halotools.mock_observables import hod_from_mock

# In[3]:


# In[4]:


from collections import Counter
def compute_occupations(halo_catalog, galaxy_catalog):
    #halo_table = cat.halocat.halo_table[cat.halocat.halo_table['halo_mvir'] > min_ptcl*cat.pmass]

    cens_occ = np.zeros((np.sum(halo_catalog['halo_upid'] == -1),))
    #cens_occ = np.zeros((len(halo_table),))
    sats_occ = np.zeros_like(cens_occ)
    print galaxy_catalog.columns
    print halo_catalog.columns
    detected_central_ids = set(galaxy_catalog[galaxy_catalog['halo_upid']==-1]['halo_id'])
    detected_satellite_upids = Counter(galaxy_catalog[galaxy_catalog['halo_upid']!=-1]['halo_upid'])

    for idx, row  in enumerate(halo_catalog[halo_catalog['halo_upid'] == -1]):
        if idx%1000000 == 0:
            print idx
            
        cens_occ[idx] = 1.0 if row['halo_id'] in detected_central_ids else 0.0
        sats_occ[idx]+= detected_satellite_upids[row['halo_id']]

    return cens_occ, sats_occ


# In[5]:


from math import ceil
def compute_mass_bins(prim_haloprop, dlog10_prim_haloprop=0.05):   
    lg10_min_prim_haloprop = np.log10(np.min(prim_haloprop))-0.001
    lg10_max_prim_haloprop = np.log10(np.max(prim_haloprop))+0.001
    num_prim_haloprop_bins = (lg10_max_prim_haloprop-lg10_min_prim_haloprop)/dlog10_prim_haloprop
    return np.logspace(
        lg10_min_prim_haloprop, lg10_max_prim_haloprop,
        num=int(ceil(num_prim_haloprop_bins)))


# In[6]:


from glob import glob
#hdf5_files = glob('../*catalog_ab_halo*fixed*.hdf5')
scratch_path = '/home/users/swmclau2/scratch/'
hdf5_files = ['catalog_ab_halo_mpeak_shuffled.hdf5', 'catalog_ab_halo_vmax@mpeak.hdf5','catalog_ab_halo_mpeak.hdf5']
for fname in hdf5_files:
    f = h5py.File(scratch_path+ fname, "r")
    print fname
    print '*'*50
    f.close()


# In[ ]:


paths = ['halo_mpeak_shuffled', 'halo_vmax@mpeak_catalog', 'halo_mpeak_catalog']


# In[ ]:


Lbox = 1000.0
#catalog = np.loadtxt('ab_sham_hod_data_cut.npy')
sim = ''
cen_hods = []
sat_hods = []
halo_catalog = astropy.table.Table.read(scratch_path+'catalog_ab_%s_large.hdf5'%('halo_mpeak'), format = 'hdf5')
    
mass_bins = compute_mass_bins(halo_catalog['halo_mvir'], 0.2)
#mass_bins = np.loadtxt('mass_bins.npy')
mass_bin_centers = (mass_bins[1:]+mass_bins[:-1])/2.0
simname = 'ds_' if sim=='' else sim
np.savetxt(simname+'mass_bins.npy', mass_bins)

for fname, path in zip(hdf5_files, paths):
    print path
    
    #for ab_property in ('halo_vmax@mpeak', 'halo_mpeak'):
        
    #   for shuffle in (False, True):
            
    #        if shuffle and ab_property=='halo_vmax@mpeak':
    #            continue
    #        if not shuffle:
    #            galaxy_catalog = astropy.table.Table.read('../%scatalog_ab_%s_fixed.hdf5'%(sim,ab_property), format = 'hdf5',
    #                                                     path = '%scatalog_ab_%s.hdf5'%(sim,ab_property))
    #        else:
    galaxy_catalog = astropy.table.Table.read(scratch_path + fname, path = path,  format = 'hdf5')

    cens_occ, sats_occ = compute_occupations(halo_catalog, galaxy_catalog)

# TODO not sure what these were for... is mvir correct?
    #host_halo_mass = halo_catalog[ halo_catalog['halo_upid']==-1]['halo_mvir']
    #host_halo_masses.append(host_halo_mass)

    cenmask = galaxy_catalog['halo_upid']==-1
    satmask = galaxy_catalog['halo_upid']>0

    halo_mass = halo_catalog['halo_mvir']

    cen_hod = hod_from_mock(galaxy_catalog['halo_mvir_host_halo'][cenmask], halo_mass, mass_bins)[0]
    sat_hod = hod_from_mock(galaxy_catalog['halo_mvir_host_halo'][satmask], halo_mass, mass_bins)[0]
    
    cen_hods.append(cen_hod)
    sat_hods.append(sat_hod)

    #if not shuffle:
    np.savetxt('catalog_ab_%s_cen_hod.npy'%(path), cen_hod)
    np.savetxt('catalog_ab_%s_sat_hod.npy'%(path),sat_hod)
    #else:
    #    np.savetxt('%scatalog_ab_%s_shuffled_cen_hod.npy'%(sim,ab_property), cen_hod)
    #    np.savetxt('%scatalog_ab_%s_shuffled_sat_hod.npy'%(sim,ab_property),sat_hod)



