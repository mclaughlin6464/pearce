#!/.conda/envs/hodemulator/bin/python
from pearce.emulator import make_training_data
from pearce.emulator import DEFAULT_PARAMS as ordered_params
ordered_params['f_c'] = (0.95, 1.0) #I don't have a way to fixed this yet...
ordered_params['logMmin'] = (12.5, 13.5)
ordered_params['sigma_logM'] = (0.2, 1.0)

make_training_data('/u/ki/swmclau2/Git/pearce/bin/training_config.cfg', \
                    ordered_params = ordered_params)
