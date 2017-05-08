#!/.conda/envs/hodemulator/bin/python
from pearce.emulator import make_training_data, parameter, DEFAULT_PARAMS

or_params = DEFAULT_PARAMS

or_params.extend([parameter('mean_occupation_centrals_assembias_param1', -1.0, 1.0),
                  parameter('disp_func_slope_centrals', -3.0, 3.0),
                  parameter('mean_occupation_satellites_assembias_param1', -1.0, 1.0),
                  parameter('disp_func_slope_satellites', -3.0, 3.0) ])

make_training_data('/u/ki/swmclau2/Git/pearce/bin/training_config.cfg', or_params)


