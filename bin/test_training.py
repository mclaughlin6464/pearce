#!/.conda/envs/hodemulator/bin/python
from pearce.emulator import make_training_data, parameter, DEFAULT_PARAMS

or_params = DEFAULT_PARAMS

or_params.extend([parameter('mean_occupation_centrals_assembias_param1', -1.0, 1.0),
                  parameter('disp_func_slope_centrals', 1e-3, 1e3),
                  parameter('mean_occupation_satellites_assembias_param1', -1.0, 1.0),
                  parameter('disp_func_slope_satellites', 1e-3, 1e3) ])

make_training_data('/u/ki/swmclau2/Git/pearce/bin/training_config.cfg', or_params)


