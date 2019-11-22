#!/.conda/envs/hodemulator/bin/python
from pearce.mocks.kittens import cat_dict
#0.65796
#0.74993
#0.85474
#0.8112
#cat = cat_dict['ds_14_b'](scale_factors = [0.538461538462], system = 'sherlock')
#cat = cat_dict['chinchilla'](400.0, scale_factors = [1.0], system = 'sherlock')
#cat = cat_dict['ds_14_b'](scale_factors = [0.0], system = 'sherlock')
#cat = cat_dict['multidark_highres'](scale_factors = [1.00110], system = 'sherlock')
#cat.cache(overwrite = True, add_local_density=False)#, add_particles=True,downsample_factor = 1e-3)

#for boxno in xrange(7):
#    for realization in xrange(1):
#for boxno in xrange(19,40):
     #cat = cat_dict['testbox'](boxno, realization,scale_factors = [1.0], system = 'ki-ls')
boxno = 0
cat = cat_dict['trainingbox'](boxno, scale_factors = [1.0], system = 'ki-ls')
    #cat = cat_dict['resolution'](boxno, scale_factors = [0.8, 1.0], system = 'ki-ls')

cat.cache(overwrite = True, add_local_density=True, add_particles = True, downsample_factor = 1e-2)
