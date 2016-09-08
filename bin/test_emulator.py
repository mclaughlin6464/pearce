#!/.conda/envs/hodemulator/bin/python
from pearce.emulator.emu import OriginalRecipe, ExtraCrispy
from pearce.emulator.trainingData import parameter, PARAMS
import numpy as np
training_dir = '/u/ki/swmclau2/des/PearceTraining/'
fiducial_params = {'logM0': 12.20, 'logM1': 13.7, 'alpha': 1.02,
                      'logMmin': 12.1, 'f_c': 0.19, 'sigma_logM': 0.46}
rbins = np.array([  0.06309573,   0.12437607,   0.24517359,   0.34422476, 0.48329302, 0.67854546,\
           0.9526807 , 1.33756775,1.8779508 ,   2.6366509 ,   3.70186906,   5.19743987, 7.29722764, \
           10.24533859,  14.38449888,  20.1958975 , 28.35512583,  39.81071706] )
rpoints = (rbins[1:]+rbins[:-1])/2

or_params = PARAMS[:]
or_params.append(parameter('r', 0, 1)) #95% sure bounds aren't used for r
emu1 = OriginalRecipe(training_dir,or_params)
#TODO test plot_data
#emu1.train()
outputs1 = emu1.emulate_wrt_r(fiducial_params, rpoints)
print outputs1

emu2 = ExtraCrispy(training_dir)#no r in params
#emu2.train()
outputs2 = emu2.emulate_wrt_r(fiducial_params, rpoints)
print outputs2
