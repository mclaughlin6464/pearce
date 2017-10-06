#!/.conda/envs/hodemulator/bin/python
from pearce.emulator import make_training_data
from pearce.emulator import DEFAULT_PARAMS as ordered_params
ordered_params['f_c'] = (0.95, 1.0) #I don't have a way to fixed this yet...

make_training_data('/u/ki/swmclau2/Git/pearce/bin/training_config.cfg', \
                    ordered_params = ordered_params)
