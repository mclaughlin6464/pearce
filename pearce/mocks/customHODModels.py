#!/bin/bash
# This module contains custom HODModels, which are subclasses of Halotools implementations.

import numpy as np
from halotools.empirical_models import Zheng07Cens, Zheng07Sats
from halotools.empirical_models import HeavisideAssembias

class RedMagicCens(Zheng07Cens):
    '''Tweak of the Zheng model to add a new parameter, f_c, denoting a modified central fraction.'''

    def __init__(self, **kwargs):
        super(RedMagicCens, self).__init__(**kwargs)

        # Default values from our analysis
        defaults = {'logMmin': 12.1, 'f_c': 0.19, 'sigma_logM': 0.46}

        self.param_dict.update(defaults)  # overwrite halotools zheng07 defaults with our own

    def mean_occupation(self, **kwargs):
        '''See Zheng07 for details.'''
        return self.param_dict['f_c'] * super(RedMagicCens, self).mean_occupation(**kwargs)

class AssembiasRedMagicCens(RedMagicCens, HeavisideAssembias):
    '''RedMagic Cens with Assembly bias'''
    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasRedMagicCens, self).__init__(**kwargs)
        HeavisideAssembias.__init__(self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate='mean_occupation', **kwargs)

class RedMagicSats(Zheng07Sats):
    '''Tweak of Zheng model to add a new parameter, f_c, denoting a modified central fraction.'''

    def __init__(self, cenocc_model, **kwargs):
        # We define modulation to be true. An existing central model is required.
        super(RedMagicSats, self).__init__(modulate_with_cenocc=True, cenocc_model=cenocc_model, **kwargs)
        defaults = {'logM0': 12.20, 'logM1': 13.7, 'alpha': 1.02, 'logMmin': 12.1, 'f_c': 0.19, 'sigma_logM': 0.46}
        self.param_dict.update(defaults)
        # It does not like that we have parameters defined multiple places.
        # Required for the central occupations.
        self._suppress_repeated_param_warning = True

    def mean_occupation(self, **kwargs):
        "See Zheng07 for details"
        f_c = 1
        if 'f_c' in self.param_dict:
            f_c = self.param_dict['f_c']

        return super(RedMagicSats, self).mean_occupation(**kwargs) / f_c

class AssembiasRedMagicSats(RedMagicSats, HeavisideAssembias):
    '''RedMagic Cens with Assembly bias'''
    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasRedMagicSats, self).__init__(**kwargs)
        HeavisideAssembias.__init__(self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate='mean_occupation', **kwargs)


class StepFuncCens(Zheng07Cens):
    '''HOD model mainly for test purposes; a step function in centrals.'''

    def __init__(self, **kwargs):
        super(StepFuncCens, self).__init__(**kwargs)
        self.param_dict['logMmin'] = 12.1  # set default

    def mean_occupation(self, **kwargs):
        """See Zheng07 for details"""
        if 'table' in kwargs.keys():
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``StepFuncCens`` class.\n")
            raise RuntimeError(msg)
        mass = np.array(mass)
        if np.shape(mass) == ():
            mass = np.array([mass])

        Mmin = 10 ** self.param_dict['logMmin']

        return np.array(mass > Mmin, dtype=int)


class StepFuncSats(Zheng07Sats):
    """HOD model mainly for testing that is 0 for satellites."""

    def mean_occupation(self, **kwargs):
        """see Zheng07 for details."""
        "See Zheng07 for details"
        if 'table' in kwargs.keys():
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``StepFuncCens`` class.\n")
            raise RuntimeError(msg)
        mass = np.array(mass)
        if np.shape(mass) == ():
            mass = np.array([mass])

        return np.zeros_like(mass)
