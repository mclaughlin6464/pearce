
# coding: utf-8 
# This notebook will make abundance matched catalogs for Jeremy and Zhongxu. I'm gonna send this notebook along to them as well in case there's something not quite right that they want to adjust. The catalogs requested were defined as follows:
# - 2 catalogs, M_peak and V_max @ M_peak (which is how, I believe, V_max is defined in this catalog. Will check tho).
# - scatter 0.18 dex
# - number density of 4.2e-4
# - z = 0.55
# - using the SMF Jeremy provided, which is in this directory with name DR10_cBOSS_WISE_SMF_z0.45_0.60_M7.dat
# - On DS14, which is located on ki-ls at /nfs/slac/des/fs1/g/sims/yymao/ds14_b_sub, courtesy of Yao
# - Include in the catalog, along with the galaxies, M_vir, x, y, z, vx, vy, vz, M_gal, am_i_a_satellite?, and M_host

from os import path
import numpy as np
from AbundanceMatching import *
from halotools.sim_manager import RockstarHlistReader, CachedHaloCatalog

halo_dir = '/scratch/users/swmclau2/MDPL2/'
#halo_dir = '/nfs/slac/des/fs1/g/sims/yymao/ds14_b_sub/hlists/'
#halo_dir = '/scratch/users/swmclau2/hlists/ds_14_b_sub/hlists/'
a = 1.0#0.65
z = 1.0/a - 1 # ~ 0.55
fname = path.join(halo_dir,  'hlist_%.5f.list'%a)

columns_to_keep = {'halo_id': (1, 'i8'), 'halo_upid':(6,'i8'), 'halo_mvir':(10, 'f4'), 'halo_x':(17, 'f4'),                        'halo_y':(18,'f4'), 'halo_z':(19,'f4'),'halo_vx':(20,'f4'), 'halo_vy':(21, 'f4'), 'halo_vz':(22,'f4'),
                  'halo_rvir': (11, 'f4'),'halo_rs':(12,'f4'), 'halo_mpeak':(58, 'f4'),'halo_vmax@mpeak':(72, 'f4'), 'halo_m200b':(39, 'f4')}


simname = 'mdpl2'


# Only run the below if you want to cache, which is useful maybe the first time (maybe). It takes ~30 min and some disk space, so be warned.
# 
# Update (Feb 1st, 2019): I had to edit halotools to make this work. The last line of the halocat was missing values... Specifically making the reader stop iteration once it encountered an indexerror. 
#reader = RockstarHlistReader(fname, columns_to_keep, '/scratch/users/swmclau2/halocats/hlist_%.2f.list.%s.hdf5'%(a, simname),\
#                             simname,'rockstar', z, 'default', 1000.0, 2.44e9, overwrite=True, header_char = '#')
#reader.read_halocat(['halo_rvir', 'halo_rs'], write_to_disk=False, update_cache_log=False)
#
#reader.add_supplementary_halocat_columns()
#reader.write_to_disk()
#reader.update_cache_log()
# In[15]:


halocat = CachedHaloCatalog(simname = simname, halo_finder='rockstar', redshift = z,version_name='most_recent')

print halocat.halo_table.colnames
# In[18]:


#### TMP ####
# Gonna do a preliminary mass cut on the halocatalog
n_part = 20
# TODO do with mpeak instead
pmass = 1.5e9
halo_table = halocat.halo_table[halocat.halo_table['halo_mvir']>n_part*pmass]

smf = np.genfromtxt('/home/users/swmclau2/Git/pearce/bin/shams/smf_dr72bright34_m7_lowm.dat', skip_header=True)[:,0:2]

#smf = np.genfromtxt('/scratch/users/swmclau2/smf_dr72bright34_m7_lowm.dat', skip_header=True)[:,0:2]
#smf = np.genfromtxt('DR10_cBOSS_WISE_SMF_z0.45_0.60_M7.dat', skip_header=True)[:,0:2]


# In[19]:


# In[20]:


nd = 5e-4#4.2e-4 #nd of final cat 


# In[21]:


#ab_property = 'halo_mpeak'
ab_property = 'halo_mvir'
#ab_property = 'halo_vmax@mpeak'
#ab_property = 'halo_vmax'
#ab_property = 'halo_vpeak'


# In[22]:

af = AbundanceFunction(smf[:,0], smf[:,1], (9.0, 12.9), faint_end_first = True)

scatter = 0.2#0.15
remainder = af.deconvolute(scatter, 20)


# In[ ]:


nd_halos = calc_number_densities(halo_table[ab_property], 1000.0) #don't think this matters which one i choose here


# In[ ]:


#check the abundance function
# In[ ]:


catalog = af.match(nd_halos, scatter)


# In[ ]:


# In[ ]:
h = 0.674 
n_obj_needed = int(nd*((1000.0)**3)) # don't divide by h


# In[ ]:

non_nan_idxs = ~np.isnan(catalog)
sort_idxs = np.argsort(catalog[non_nan_idxs])[::-1]
final_catalog = catalog[non_nan_idxs][sort_idxs][:n_obj_needed]

output = halo_table[non_nan_idxs][sort_idxs][:n_obj_needed]


output['gal_smass'] = final_catalog



#output.write('/nfs/slac/g/ki/ki18/des/swmclau2/catalog_ab_%s_large.hdf5'%ab_property, format = 'hdf5', path = '%s_catalog'%ab_property, overwrite=True)
#output.write('/scratch/users/swmclau2/test_MDPL2_%s_smf_sham_large.hdf5'%ab_property, format = 'hdf5', path = '%s_catalog'%ab_property, overwrite=True)
#output.write('/scratch/users/swmclau2/MDPL2_%s_smf_sham.hdf5'%ab_property, format = 'hdf5', path = '%s_catalog'%ab_property, overwrite=True)
np.save('/scratch/users/swmclau2/UniverseMachine/cut_macc_sham_catalog.npy', output.as_array()[:n_obj_needed])
#print ab_property
