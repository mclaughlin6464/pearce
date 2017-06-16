#!/.conda/envs/hodemulator/bin/python
from pearce.emulator import make_training_data, parameter, DEFAULT_PARAMS

or_params = DEFAULT_PARAMS

or_params.update({'mean_occupation_centrals_assembias_param1':( -1.0, 1.0),
                  'disp_func_slope_centrals':(-1.0, 1.0),
                  'mean_occupation_satellites_assembias_param1':( -1.0, 1.0),
                  'disp_func_slope_satellites':(-1.0, 1.0) })

make_training_data('/u/ki/swmclau2/Git/pearce/bin/training_config.cfg', or_params)


