#!/bin/bash
# This module contains my modification of Hearin's decorated HOD, that is a continuous extension of the idea.

from itertools import izip
import inspect

import numpy as np
from halotools.empirical_models import HeavisideAssembias

def sigmoid(sec_haloprop, slope=1):
    return np.reciprocal(1+np.exp(-slope*sec_haloprop))

class ContinuousAssembias(HeavisideAssembias):
    '''
    Class used to extend the behavior of decorated assembly bias for continuous distributions.
    '''

    def __init__(self, disp_func = sigmoid, **kwargs):
        '''
        For full documentation, see the Heaviside Assembias declaration in Halotools.

        This adds one additional kwarg, the disp_func. This function determines the displacement to the mean performed
        by decoration as a function of the secondary parameter. The default is the sigmoid function, decleared above.
        :param disp_func:
            Function used to displace populations along the mean according to the secondary parameter. Default is the
            sigmoid function.
            Must accept an array of secondary statistics as first argument. The input will be shifted so the "split"
            will be at 0.
            All other arguments must be kwargs, and will be put in the param_dict with their default arguements.
        :param kwargs:
            All other arguements. For details, see the declaration of HeavisideAssembias in Halotools.
        '''
        super(ContinuousAssembias, self).__init__(**kwargs)

        self.disp_func = disp_func

    def _initialize_assembias_param_dict(self, assembias_strength=0.5, **kwargs):
        '''
        For full documentation, see the Heaviside Assembias Declaration in Halotools.

        This function calls the superclass's version. Then, it adds the parameters from disp_func to the dict as well.
        This is taken by inspecting the function signature and no input are needed.
        :param assembias_strength:
            Strength of assembias.
        :param kwargs:
            Other kwargs. Details in superclass.
        :return: None
        '''
        super(ContinuousAssembias, self)._initialize_assembias_param_dict(assembias_strength=assembias_strength, **kwargs)

        #get the function specification from the displacement function
        argspec = inspect.getargspec(self.disp_func)
        #add any additional parameters to the parameter dict
        for ipar, val in izip(argspec.args[1:], argspec.defaults):
            self.param_dict[self._get_assembias_param_dict_key(ipar)] = val

    def _galprop_perturbation(self, **kwargs):
        """
        Method determines hwo much to boost the baseline function
        according to the strength of assembly bias and the min/max
        boost allowable by the requirement that the all-halo baseline
        function be preserved. The returned perturbation applies to type-1 halos.
        :param kwargs:
            Required kwargs are:
                baseline_result
                splitting_result
                prim_haloprop
        :return: result, np.
        """