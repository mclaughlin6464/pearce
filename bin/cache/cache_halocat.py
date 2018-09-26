#!/.conda/envs/hodemulator/bin/python
from pearce.mocks.kittens import cat_dict
#0.65796
#0.74993
#0.85474
#0.8112
#cat = cat_dict['chinchilla'](400.0, scale_factors = [0.54780, 0.59260, 0.658, 0.71170, 0.8112], system = 'ki-ls')

#cat.cache(overwrite = True, add_local_density=False, add_particles=True,downsample_factor = 1e-3)

#for boxno in xrange(7):
#    for realization in xrange(1):
for boxno in xrange(40):
     #cat = cat_dict['testbox'](boxno, realization,scale_factors = [1.0], system = 'ki-ls')
    cat = cat_dict['trainingbox'](boxno, scale_factors = [1.0], system = 'sherlock')
    #cat = cat_dict['resolution'](boxno, scale_factors = [0.8, 1.0], system = 'ki-ls')

    cat.cache(overwrite = True)#, add_local_density=True, add_particles = False, downsample_factor = 1e-2)

