#!/.conda/envs/hodemulator/bin/python
from pearce.mocks.kittens import cat_dict
#0.65796
#0.74993
#0.85474
#0.8112
cat = cat_dict['chinchilla'](400.0, scale_factors = [0.658, 0.8112, 1.0], system = 'sherlock')

cat.cache(overwrite = True, add_local_density=False, add_particles=True,downsample_factor = 1e-2)

