#!/.conda/envs/hodemulator/bin/python
from pearce.emulator import make_training_data
from pearce.emulator import DEFAULT_PARAMS as ordered_params
ordered_params['f_c'] = (0.05, .5)
ordered_params['logMmin'] = (13.0, 14.5)
ordered_params['sigma_logM'] = (0.05, 1.0)
ordered_params['logM1'] = (13.0, 16.0)
ordered_params['alpha'] = (0.7, 1.2)

ordered_params.update({'mean_occupation_centrals_assembias_param1':( -1.0, 1.0),
                      'mean_occupation_satellites_assembias_param1':( -1.0, 1.0)})

make_training_data('/u/ki/swmclau2/Git/pearce/bin/training_data/training_config.cfg',ordered_params)


