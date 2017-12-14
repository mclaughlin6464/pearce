#!/.conda/envs/hodemulator/bin/python
from pearce.emulator import make_training_data
from pearce.emulator import DEFAULT_PARAMS as ordered_params

ordered_params = {'mean_occupation_centrals_assembias_param1':( -1.0, 1.0),
                  'mean_occupation_centrals_assembias_split1':( 0.0, 1.0),
                  'mean_occupation_satellites_assembias_param1':( -1.0, 1.0),
                  'mean_occupation_satellites_assembias_split1':( 0.0, 1.0)}

make_training_data('/u/ki/swmclau2/Git/pearce/bin/tabulated_training_assembias.cfg',ordered_params)


