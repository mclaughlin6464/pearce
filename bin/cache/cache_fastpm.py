#!/.conda/envs/hodemulator/bin/python
from pearce.mocks.kittens import cat_dict

for boxno in xrange(122):
    print boxno
    cat = cat_dict['fastpm'](boxno, scale_factors = [0.645], system = 'ki-ls')
    cat.cache(overwrite = True)
