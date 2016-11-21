#!/.conda/envs/hodemulator/bin/python
from pearce.mocks.kittens import cat_dict

cat = cat_dict['chinchilla'](400.0, scale_factors = [0.658, 1.0])

cat.cache(scale_factors=[0.658], overwrite = True, add_local_density=True)

