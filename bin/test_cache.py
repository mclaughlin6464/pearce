#!/.conda/envs/hodemulator/bin/python
from pearce.mocks.kittens import cat_dict
#0.65796
#0.74993
#0.85474
cat = cat_dict['chinchilla'](400.0, scale_factors = [0.658, 0.749,0.854, 1.0])

cat.cache(overwrite = True, add_local_density=True)

