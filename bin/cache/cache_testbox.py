#!/.conda/envs/hodemulator/bin/python
from pearce.mocks.kittens import cat_dict

for boxno in xrange(7):
    for realization in xrange(5):
        #cat = cat_dict['testbox'](boxno, scale_factors = [0.25, 0.333, 0.5, 0.540541, 0.588235, 0.645161, 0.714286, 0.8, 0.909091, 1.0], system = 'sherlock')
        cat = cat_dict['testbox'](boxno, realization , scale_factors = [1.0], system = 'ki-ls')


        cat.cache(overwrite = True, add_local_density=False, add_particles = True, downsample_factor = 1e-2)

