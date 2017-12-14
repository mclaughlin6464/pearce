#!/.conda/envs/hodemulator/bin/python
from pearce.emulator import make_training_data
from pearce.emulator import DEFAULT_PARAMS as ordered_params
ordered_params['f_c'] = (0.05, .5)
ordered_params['logMmin'] = (12.5, 13.5)
ordered_params['sigma_logM'] = (0.2, 1.0)
ordered_params['logM1'] = (13.0, 16.0)
ordered_params['alpha'] = (0.7, 1.2)

make_training_data('/u/ki/swmclau2/Git/pearce/bin/training_data/training_config.cfg',ordered_params)


