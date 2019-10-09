"""
I'm modifying the halotools tpcf code to add a few more efficincies.

One directly returns the correlation functions from subregions, so I can compute arbitary jackknifes more efficiently

Another is to add a flag for do_auto1 and do_auto2. Sometimes, you wanna compute xi_gg and xi_gm but not xi_mm!
"""

from __future__ import absolute_import, division, unicode_literals
import numpy as np
from halotools.mock_observables import *
from halotools.mock_observables.two_point_clustering import *
from halotools.mock_observables.two_point_clustering.tpcf import _tpcf_process_args, _random_counts
from halotools.mock_observables.two_point_clustering.tpcf_estimators import _TP_estimator, _TP_estimator_requirements
from halotools.mock_observables.pair_counters import npairs_jackknife_3d
from halotools.mock_observables.two_point_clustering.clustering_helpers import (process_optional_input_sample2,
    verify_tpcf_estimator, tpcf_estimator_dd_dr_rr_requirements)

from halotools.mock_observables.two_point_clustering.tpcf_jackknife import \
    _tpcf_jackknife_process_args,_enclose_in_box, get_subvolume_numbers, jrandom_counts

np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


__all__ = ['tpcf_subregions', 'tpcf']

# all lifted from duncan's code
def tpcf_subregions(sample1, randoms, rbins, Nsub=[5, 5, 5],
        sample2=None, period=None, do_auto1=True, do_auto2=False, do_cross=True,
                estimator='Natural', num_threads=1, seed=None, RR=None):

    do_auto = do_auto1 or do_auto2
# process input parameters
    function_args = (sample1, randoms, rbins, Nsub, sample2, period, do_auto,
        do_cross, estimator, num_threads, seed)
    sample1, rbins, Nsub, sample2, randoms, period, do_auto, do_cross, num_threads,\
        _sample1_is_sample2, PBCs = _tpcf_jackknife_process_args(*function_args)

# determine box size the data occupies.
# This is used in determining jackknife samples.
    if PBCs is False:
        sample1, sample2, randoms, Lbox = _enclose_in_box(sample1, sample2, randoms)
    else:
        Lbox = period


    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)

    N1 = len(sample1)
    N2 = len(sample2)
    NR = len(randoms)


    j_index_1, N_sub_vol = cuboid_subvolume_labels(sample1, Nsub, Lbox)
    j_index_2, N_sub_vol = cuboid_subvolume_labels(sample2, Nsub, Lbox)
    j_index_random, N_sub_vol = cuboid_subvolume_labels(randoms, Nsub, Lbox)

# number of points in each subvolume
    NR_subs = get_subvolume_numbers(j_index_random, N_sub_vol)
    N1_subs = get_subvolume_numbers(j_index_1, N_sub_vol)
    N2_subs = get_subvolume_numbers(j_index_2, N_sub_vol)
# number of points in each jackknife sample
    N1_subs = N1 - N1_subs
    N2_subs = N2 - N2_subs
    NR_subs = NR - NR_subs


# calculate all the pair counts
# TODO need to modify this function
    D1D1, D1D2, D2D2 = jnpair_counts(
        sample1, sample2, j_index_1, j_index_2, N_sub_vol,
            rbins, period, num_threads, do_auto1, do_cross, do_auto2, _sample1_is_sample2)
# pull out the full and sub sample results

    if _sample1_is_sample2:
        D1D1_full = D1D1[0, :]
        D1D1_sub = D1D1[1:, :]
        D1D2_full = D1D2[0, :]
        D1D2_sub = D1D2[1:, :]
        D2D2_full = D2D2[0, :]
        D2D2_sub = D2D2[1:, :]

    else:
        if do_auto1:
            D1D1_full = D1D1[0, :]
            D1D1_sub = D1D1[1:, :]
        if do_cross:
            D1D2_full = D1D2[0, :]
            D1D2_sub = D1D2[1:, :]
        if do_auto2:
            D2D2_full = D2D2[0, :]
            D2D2_sub = D2D2[1:, :]

# do random counts
# TODO figure out what of this i can skip?  

    if RR is None:
        D1R, RR = jrandom_counts(sample1, randoms, j_index_1, j_index_random, N_sub_vol,
            rbins, period, 1, do_DR, do_RR)
    else: #use the precomputed RR
        D1R, RR_dummy= jrandom_counts(sample1, randoms, j_index_1, j_index_random, N_sub_vol,
            rbins, period, 1, do_DR, do_RR=False)

    print 'A'
    if _sample1_is_sample2:
        D2R = D1R
    else:
        if do_DR is True:
            D2R, RR_dummy = jrandom_counts(sample2, randoms, j_index_2, j_index_random,
                    N_sub_vol, rbins, period, num_threads, do_DR, do_RR=False)
        else:
            D2R = None
    print 'B'
    if do_DR is True:
        D1R_full = D1R[0, :]
        D1R_sub = D1R[1:, :]
        D2R_full = D2R[0, :]
        D2R_sub = D2R[1:, :]
    else:
        D1R_full = None
        D1R_sub = None
        D2R_full = None
        D2R_sub = None
    if do_RR is True:
        RR_full = RR[0, :]
        RR_sub = RR[1:, :]
    else:
        RR_full = None
        RR_sub = None
# calculate the correlation function for the subsamples
    outputs = []
    print 'C'

    if do_auto1 or _sample1_is_sample2:
        xi_11_sub = _TP_estimator(D1D1_sub, D1R_sub, RR_sub, N1_subs, N1_subs, NR_subs, NR_subs, estimator)
        outputs.append(xi_11_sub)
    if do_cross:
        xi_12_sub = _TP_estimator(D1D2_sub, D1R_sub, RR_sub, N1_subs, N2_subs, NR_subs, NR_subs, estimator)
        outputs.append(xi_12_sub)
    if do_auto2:
        xi_22_sub = _TP_estimator(D2D2_sub, D2R_sub, RR_sub, N2_subs, N2_subs, NR_subs, NR_subs, estimator)
        outputs.append(xi_22_sub)
    return outputs[0] if len(outputs) ==1 else tuple(outputs)

def tpcf(sample1, rbins, sample2=None, randoms=None, period=None,
        do_auto1=True, do_cross=True, do_auto2=False, estimator='Natural', num_threads=1,
        approx_cell1_size=None, approx_cell2_size=None, approx_cellran_size=None,
        RR_precomputed=None, NR_precomputed=None, seed=None, n_split = 1):
    r"""
    Calculate the real space two-point correlation function, :math:`\xi(r)`.
    Example calls to this function appear in the documentation below.
    See the :ref:`mock_obs_pos_formatting` documentation page for
    instructions on how to transform your coordinate position arrays into the
    format accepted by the ``sample1`` and ``sample2`` arguments.
    See also :ref:`galaxy_catalog_analysis_tutorial2` for example usage on a
    mock galaxy catalog.
    Parameters
    ----------
    sample1 : array_like
        Npts1 x 3 numpy array containing 3-D positions of points.
        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        Examples section below, for instructions on how to transform
        your coordinate position arrays into the
        format accepted by the ``sample1`` and ``sample2`` arguments.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.
    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are counted.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.
    sample2 : array_like, optional
        Npts2 x 3 array containing 3-D positions of points.
        Passing ``sample2`` as an input permits the calculation of
        the cross-correlation function.
        Default is None, in which case only the
        auto-correlation function will be calculated.
    randoms : array_like, optional
        Nran x 3 array containing 3-D positions of randomly distributed points.
        If no randoms are provided (the default option),
        calculation of the tpcf can proceed using analytical randoms
        (only valid for periodic boundary conditions).
    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        If set to None (the default option), PBCs are set to infinity,
        in which case ``randoms`` must be provided.
        Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.
    do_auto : boolean, optional
        Boolean determines whether the auto-correlation function will
        be calculated and returned. Default is True.
    do_cross : boolean, optional
        Boolean determines whether the cross-correlation function will
        be calculated and returned. Only relevant when ``sample2`` is also provided.
        Default is True for the case where ``sample2`` is provided, otherwise False.
    estimator : string, optional
        Statistical estimator for the tpcf.
        Options are 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
        Default is ``Natural``.
    num_threads : int, optional
        Number of threads to use in calculation, where parallelization is performed
        using the python ``multiprocessing`` module. Default is 1 for a purely serial
        calculation, in which case a multiprocessing Pool object will
        never be instantiated. A string 'max' may be used to indicate that
        the pair counters should use all available cores on the machine.
    approx_cell1_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by how points
        will be apportioned into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use Lbox/10 in each dimension,
        which will return reasonable result performance for most use-cases.
        Performance can vary sensitively with this parameter, so it is highly
        recommended that you experiment with this parameter when carrying out
        performance-critical calculations.
    approx_cell2_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for sample2.  See comments for
        ``approx_cell1_size`` for details.
    approx_cellran_size : array_like, optional
        Analogous to ``approx_cell1_size``, but for randoms.  See comments for
        ``approx_cell1_size`` for details.
    RR_precomputed : array_like, optional
        Array storing the number of RR-counts calculated in advance during
        a pre-processing phase. Must have the same length as *len(rbins)*.
        If the ``RR_precomputed`` argument is provided,
        you must also provide the ``NR_precomputed`` argument.
        Default is None.
    NR_precomputed : int, optional
        Number of points in the random sample used to calculate ``RR_precomputed``.
        If the ``NR_precomputed`` argument is provided,
        you must also provide the ``RR_precomputed`` argument.
        Default is None.
    seed : int, optional
        Random number seed used to randomly downsample data, if applicable.
        Default is None, in which case downsampling will be stochastic.
    Returns
    -------
    correlation_function(s) : numpy.array
        *len(rbins)-1* length array containing the correlation function :math:`\xi(r)`
        computed in each of the bins defined by input ``rbins``.
        .. math::
            1 + \xi(r) \equiv \mathrm{DD}(r) / \mathrm{RR}(r),
        If ``estimator`` is set to 'Natural'.  :math:`\mathrm{DD}(r)` is the number
        of sample pairs with separations equal to :math:`r`, calculated by the pair
        counter.  :math:`\mathrm{RR}(r)` is the number of random pairs with separations
        equal to :math:`r`, and is counted internally using "analytic randoms" if
        ``randoms`` is set to None (see notes for an explanation), otherwise it is
        calculated using the pair counter.
        If ``sample2`` is passed as input
        (and if ``sample2`` is not exactly the same as ``sample1``),
        then three arrays of length *len(rbins)-1* are returned:
        .. math::
            \xi_{11}(r), \xi_{12}(r), \xi_{22}(r),
        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and
        ``sample2``, and the autocorrelation of ``sample2``, respectively.
        If ``do_auto`` or ``do_cross`` is set to False,
        the appropriate sequence of results is returned.
    Notes
    -----
    For a higher-performance implementation of the tpcf function written in C,
    see the Corrfunc code written by Manodeep Sinha, available at
    https://github.com/manodeep/Corrfunc.
    Examples
    --------
    For demonstration purposes we calculate the `tpcf` for halos in the
    `~halotools.sim_manager.FakeSim`.
    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> x = halocat.halo_table['halo_x']
    >>> y = halocat.halo_table['halo_y']
    >>> z = halocat.halo_table['halo_z']
    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:
    >>> sample1 = np.vstack((x,y,z)).T
    Alternatively, you may use the `~halotools.mock_observables.return_xyz_formatted_array`
    convenience function for this same purpose, which provides additional wrapper
    behavior around `numpy.vstack` such as placing points into redshift-space.
    >>> rbins = np.logspace(-1, 1, 10)
    >>> xi = tpcf(sample1, rbins, period=halocat.Lbox)
    See also
    --------
    :ref:`galaxy_catalog_analysis_tutorial2`
    """

    do_auto = do_auto1 or do_auto2
    # check input arguments using clustering helper functions
    function_args = (sample1, rbins, sample2, randoms, period,
        do_auto, do_cross, estimator, num_threads,
        approx_cell1_size, approx_cell2_size, approx_cellran_size,
        RR_precomputed, NR_precomputed, seed)

    # pass arguments in, and get out processed arguments, plus some control flow variables
    (sample1, rbins, sample2, randoms, period,
        do_auto, do_cross, num_threads,
        _sample1_is_sample2, PBCs,
        RR_precomputed, NR_precomputed) = _tpcf_process_args(*function_args)

    # What needs to be done?
    do_DD, do_DR, do_RR = tpcf_estimator_dd_dr_rr_requirements[estimator]
    if RR_precomputed is not None:
        # overwrite do_RR as necessary
        do_RR = False

    # How many points are there (for normalization purposes)?
    N1 = len(sample1)
    N2 = len(sample2)
    if randoms is not None:
        NR = len(randoms)
    else:
        # set the number of randoms equal to the number of points in sample1
        # this is arbitrarily set, but must remain consistent!
        if NR_precomputed is not None:
            NR = NR_precomputed
        else:
            NR = N1

    # count data pairs
    D1D1, D1D2, D2D2 = _pair_counts(sample1, sample2, rbins, period,
        num_threads, do_auto1, do_cross, do_auto2, _sample1_is_sample2,
        approx_cell1_size, approx_cell2_size)

    # count random pairs
    # split this up over a few because randoms is large
    # TODO do they stack like this?
    split_randoms = np.array_split(randoms, n_split, axis = 0)

    D1R, D2R, RR = np.zeros_like(D1D1), np.zeros_like(D1D2), np.zeros_like(D1D2)
    #D1Rs = []
    for i, _rand in enumerate(split_randoms):
        print i,
        _D1R, _D2R, _RR, = _random_counts(sample1, sample2, _rand, rbins,
            period, PBCs, num_threads, do_RR, do_DR, _sample1_is_sample2,
            approx_cell1_size, approx_cell2_size, approx_cellran_size)
        #D1Rs.append(_D1R)

        if _D1R is not None:
            D1R+=_D1R
        if _D2R is not None:
            D2R+=_D2R
        if _RR is not None:
            RR+=_RR

    print D1R
    D1R=np.array(D1R)/n_split
    D2R=np.array(D2R)/n_split
    RR=np.array(RR)/n_split
    print D1R
    print 

    if RR_precomputed is not None:

        RR = RR_precomputed

    # run results through the estimator and return relavent/user specified results.

    outputs = []
    if do_auto1 or _sample1_is_sample2:
        xi_11 = _TP_estimator(D1D1, D1R, RR, N1, N1, NR, NR, estimator)
        outputs.append(xi_11)
    if do_cross:
        xi_12 = _TP_estimator(D1D2, D1R, RR, N1, N2, NR, NR, estimator)
        outputs.append(xi_12)
    if do_auto2:
        xi_22 = _TP_estimator(D2D2, D2R, RR, N2, N2, NR, NR, estimator)
        outputs.append(xi_22)

    return outputs[0] if len(outputs) ==1 else tuple(outputs)


# overload to skip the xi_mm calculation
def jnpair_counts(sample1, sample2, j_index_1, j_index_2, N_sub_vol, rbins,
        period, num_threads, do_auto1 = True, do_cross=False,do_auto2=False, _sample1_is_sample2=False):
    """
    Count jackknife data pairs: DD
    """
    if do_auto1 is True:
        D1D1 = npairs_jackknife_3d(sample1, sample1, rbins, period=period,
            jtags1=j_index_1, jtags2=j_index_1,  N_samples=N_sub_vol,
            num_threads=num_threads)
        D1D1 = np.diff(D1D1, axis=1)
    else:
        D1D1 = None
        D2D2 = None

    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = npairs_jackknife_3d(sample1, sample2, rbins, period=period,
                jtags1=j_index_1, jtags2=j_index_2,
                N_samples=N_sub_vol, num_threads=num_threads)
            D1D2 = np.diff(D1D2, axis=1)
        else:
            D1D2 = None
        if do_auto2 is True:
            D2D2 = npairs_jackknife_3d(sample2, sample2, rbins, period=period,
                jtags1=j_index_2, jtags2=j_index_2,
                N_samples=N_sub_vol, num_threads=num_threads)
            D2D2 = np.diff(D2D2, axis=1)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2

def _pair_counts(sample1, sample2, rbins,
        period, num_threads, do_auto1, do_cross, do_auto2,
        _sample1_is_sample2, approx_cell1_size, approx_cell2_size):
    r"""
    Internal function used calculate DD-pairs during the calculation of the tpcf.
    """
    if do_auto1 is True:
        D1D1 = npairs_3d(sample1, sample1, rbins, period=period,
            num_threads=num_threads,
            approx_cell1_size=approx_cell1_size,
            approx_cell2_size=approx_cell1_size)
        D1D1 = np.diff(D1D1)
    else:
        D1D1 = None
        D2D2 = None

    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = npairs_3d(sample1, sample2, rbins, period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell1_size,
                approx_cell2_size=approx_cell2_size)
            D1D2 = np.diff(D1D2)
        else:
            D1D2 = None
        if do_auto2 is True:
            D2D2 = npairs_3d(sample2, sample2, rbins, period=period,
                num_threads=num_threads,
                approx_cell1_size=approx_cell2_size,
                approx_cell2_size=approx_cell2_size)
            D2D2 = np.diff(D2D2)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2
