#!/bin/bash
# This module contains my modification of Hearin's decorated HOD, that is a continuous extension of the idea.

from itertools import izip
import inspect
from warnings import warn

import numpy as np
from halotools.empirical_models import HeavisideAssembias
from halotools.custom_exceptions import HalotoolsError

def sigmoid(sec_haloprop, slope=1):
    return np.reciprocal(1+np.exp(-slope*sec_haloprop))

def compute_conditional_averages(disp_func = lambda x: x, **kwargs):
    """
    In bins of the ``prim_haloprop``, compute the average value of disp_func given
    the input ``table`` based on the value of ``sec_haloprop``.
    Parameters
    ----------
    disp_func: function, optional
        A kwarg that is the function to calculate the conditional average of.
        Default is 'lambda x: x' which will compute the average value of sec_haloprop
        in bins of prim_haloprop
    table : astropy table, optional
        a keyword argument that stores halo catalog being used to make mock galaxy population
        If a `table` is passed, the `prim_haloprop_key` and `sec_haloprop_key` keys
        must also be passed. If not passing a `table`, you must directly pass the
        `prim_haloprop` and `sec_haloprop` keyword arguments.
    prim_haloprop_key : string, optional
        Name of the column of the input ``table`` that will be used to access the
        primary halo property. `compute_conditional_percentiles` bins the ``table`` by
        ``prim_haloprop_key`` when computing the result.
    sec_haloprop_key : string, optional
        Name of the column of the input ``table`` that will be used to access the
        secondary halo property. `compute_conditional_percentiles` bins the ``table`` by
        ``prim_haloprop_key``, and in each bin uses the value stored in ``sec_haloprop_key``
        to compute the ``prim_haloprop``-conditioned rank-order percentile.
    prim_haloprop : array_like, optional
        Array storing the primary halo property used to bin the input points.
        If a `prim_haloprop` is passed, you must also pass a `sec_haloprop`.
    sec_haloprop : array_like, optional
        Array storing the secondary halo property used to define the conditional percentiles
        in each bin of `prim_haloprop`.
    prim_haloprop_bin_boundaries : array, optional
        Array defining the boundaries by which we will bin the input ``table``.
        Default is None, in which case the binning will be automatically determined using
        the ``dlog10_prim_haloprop`` keyword.
    dlog10_prim_haloprop : float, optional
        Logarithmic spacing of bins of the mass-like variable within which
        we will assign secondary property percentiles. Default is 0.2.
    Examples
    --------
    >>> from halotools.sim_manager import FakeSim
    >>> fakesim = FakeSim()
    >>> result = compute_conditional_percentiles(table = fakesim.halo_table, prim_haloprop_key = 'halo_mvir', sec_haloprop_key = 'halo_vmax')
    Notes
    -----
    The sign of the result is such that in bins of the primary property,
    *smaller* values of the secondary property
    receive *smaller* values of the returned percentile.
    """

    if 'table' in kwargs:
        table = kwargs['table']
        try:
            prim_haloprop_key = kwargs['prim_haloprop_key']
            prim_haloprop = table[prim_haloprop_key]
            sec_haloprop_key = kwargs['sec_haloprop_key']
            sec_haloprop = table[sec_haloprop_key]
        except KeyError:
            msg = ("\nWhen passing an input ``table`` to the ``compute_conditional_percentiles`` method,\n"
                "you must also pass ``prim_haloprop_key`` and ``sec_haloprop_key`` keyword arguments\n"
                "whose values are column keys of the input ``table``\n")
            raise HalotoolsError(msg)
    else:
        try:
            prim_haloprop = kwargs['prim_haloprop']
            sec_haloprop = kwargs['sec_haloprop']
        except KeyError:
            msg = ("\nIf not passing an input ``table`` to the ``compute_conditional_percentiles`` method,\n"
                "you must pass a ``prim_haloprop`` and ``sec_haloprop`` arguments\n")
            raise HalotoolsError(msg)

    def compute_prim_haloprop_bins(dlog10_prim_haloprop=0.05, **kwargs):
        """
        Parameters
        ----------
        prim_haloprop : array
            Array storing the value of the primary halo property column of the ``table``
            passed to ``compute_conditional_percentiles``.
        prim_haloprop_bin_boundaries : array, optional
            Array defining the boundaries by which we will bin the input ``table``.
            Default is None, in which case the binning will be automatically determined using
            the ``dlog10_prim_haloprop`` keyword.
        dlog10_prim_haloprop : float, optional
            Logarithmic spacing of bins of the mass-like variable within which
            we will assign secondary property percentiles. Default is 0.05.
        Returns
        --------
        output : array
            Numpy array of integers storing the bin index of the prim_haloprop bin
            to which each halo in the input table was assigned.
        """
        try:
            prim_haloprop = kwargs['prim_haloprop']
        except KeyError:
            msg = ("The ``compute_prim_haloprop_bins`` method "
                "requires the ``prim_haloprop`` keyword argument")
            raise HalotoolsError(msg)

        try:
            prim_haloprop_bin_boundaries = kwargs['prim_haloprop_bin_boundaries']
        except KeyError:
            lg10_min_prim_haloprop = np.log10(np.min(prim_haloprop))-0.001
            lg10_max_prim_haloprop = np.log10(np.max(prim_haloprop))+0.001
            num_prim_haloprop_bins = (lg10_max_prim_haloprop-lg10_min_prim_haloprop)/dlog10_prim_haloprop
            prim_haloprop_bin_boundaries = np.logspace(
                lg10_min_prim_haloprop, lg10_max_prim_haloprop,
                num=ceil(num_prim_haloprop_bins))

        # digitize the masses so that we can access them bin-wise
        output = np.digitize(prim_haloprop, prim_haloprop_bin_boundaries)

        # Use the largest bin for any points larger than the largest bin boundary,
        # and raise a warning if such points are found
        Nbins = len(prim_haloprop_bin_boundaries)
        if Nbins in output:
            msg = ("\n\nThe ``compute_prim_haloprop_bins`` function detected points in the \n"
                "input array of primary halo property that were larger than the largest value\n"
                "of the input ``prim_haloprop_bin_boundaries``. All such points will be assigned\n"
                "to the largest bin.\nBe sure that this is the behavior you expect for your application.\n\n")
            warn(msg)
            output = np.where(output == Nbins, Nbins-1, output)

        return output

    compute_prim_haloprop_bins_dict = {}
    compute_prim_haloprop_bins_dict['prim_haloprop'] = prim_haloprop
    try:
        compute_prim_haloprop_bins_dict['prim_haloprop_bin_boundaries'] = (
            kwargs['prim_haloprop_bin_boundaries'])
    except KeyError:
        pass
    try:
        compute_prim_haloprop_bins_dict['dlog10_prim_haloprop'] = kwargs['dlog10_prim_haloprop']
    except KeyError:
        pass
    prim_haloprop_bins = compute_prim_haloprop_bins(**compute_prim_haloprop_bins_dict)

    output = np.zeros_like(prim_haloprop)

    # sort on secondary property only with each mass bin
    bins_in_halocat = set(prim_haloprop_bins)
    for ibin in bins_in_halocat:
        indices_of_prim_haloprop_bin = np.where(prim_haloprop_bins == ibin)[0]

        # place the percentiles into the catalog
        output[indices_of_prim_haloprop_bin] = np.mean(disp_func(sec_haloprop[indices_of_prim_haloprop_bin]))

    #TODO i'm not sure if this should have dimensions of prim_haloprop or the binning...
    return output


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

    #TODO testme
    #TODO this isn't done because I don't do the median subtraction.
    def _galprop_perturbation(self, **kwargs):
        """
        Method determines hwo much to boost the baseline function
        according to the strength of assembly bias and the min/max
        boost allowable by the requirement that the all-halo baseline
        function be preserved. The returned perturbation applies to type-1 halos.

        Uses the disp_func passed in duing perturbation
        :param kwargs:
            Required kwargs are:
                baseline_result
                prim_haloprop
                sec_haloprop
        :return: result, np.arry with dimensions of prim_haloprop detailing the perturbation.
        """

        #lower_bound_key = 'lower_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        #baseline_lower_bound = getattr(self, lower_bound_key)
        upper_bound_key = 'upper_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        baseline_upper_bound = getattr(self, upper_bound_key)

        try:
            baseline_result = kwargs['baseline_result']
            prim_haloprop = kwargs['prim_haloprop']
            sec_haloprop = kwargs['sec_haloprop']
        except KeyError:
            msg = ("Must call _galprop_perturbation method of the"
                   "HeavisideAssembias class with the following keyword arguments:\n"
                   "``baseline_result``, ``splitting_result`` and ``prim_haloprop``")
            raise HalotoolsError(msg)

        #evaluate my continuous modification
        strength = self.assembias_strength(prim_haloprop)

        average = compute_conditional_averages(self.disp_func, prim_haloprop=prim_haloprop, sec_haloprop=sec_haloprop)

        bound1 = baseline_result/average
        bound2 = (baseline_upper_bound - baseline_result)/(baseline_upper_bound-average)
        bound = np.minimum(bound1, bound2)

        result = strength*bound(self.disp_func(sec_haloprop)- average)

        return result