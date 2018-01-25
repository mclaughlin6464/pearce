#!/.conda/envs/hodemulator/bin/python
from pearce.mocks.kittens import cat_dict
#0.65796
#0.74993
#0.85474
#0.8112
for boxno in range(7):
    cat = cat_dict['testbox'](boxno,0, scale_factors = [0.5, 0.540541, 0.588235, 0.645161], system = 'ki-ls')

    cat.cache(overwrite = True, add_local_density=False)

