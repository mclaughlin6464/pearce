#!/bin/bash
# TODO better docstring
'''Module containing the main Cat object, which implements caching, loading, populating, and calculating observables
on catalogs. '''

from os import path
from itertools import izip
from multiprocessing import cpu_count
import warnings
from time import time
import numpy as np
from astropy import cosmology
from halotools.sim_manager import RockstarHlistReader, CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.empirical_models import HodModelFactory, TrivialPhaseSpace, NFWPhaseSpace
from halotools.mock_observables import * #i'm importing so much this is just easier
from .customHODModels import RedMagicSats, RedMagicCens, StepFuncCens, StepFuncSats

# try to import corrfunc, determine if it was successful
try:
    from Corrfunc._countpairs import counpairs_xi

    CORRFUNC_AVAILABLE = True
except ImportError:
    CORRFUNC_AVAILABLE = False

# TODO check that a model has been loaded?
def observable(func):
    '''
    Decorator for observable methods. Checks that the catalog is properly loaded and calcualted.
    :param func:
        Function that calculates an observable on a populated catalog.
    :return: func
    '''
    def _func(self, *args, **kwargs):
        try:
            assert self.halocat is not None
            assert self.model is not None
            assert self.populated_once
        except AssertionError:
            raise AssertionError("The cat must have a loaded model and catalog and be populated before calculating an observable.")
        return func(self, *args, **kwargs)

    return _func

class Cat(object):
    def __init__(self, simname='cat', loc='', filenames=[], cache_loc='/u/ki/swmclau2/des/halocats/',
                 columns_to_keep={}, halo_finder='rockstar', version_name='most_recent',
                 Lbox=1.0, pmass=1.0, scale_factors=[], cosmo=cosmology.WMAP5,
                 **kwargs):

        # TODO need both loc and filenames?
        # TODO emply filenames does a glob?
        # TODO change 'loc' to dir?
        '''
        The main object controlling manipulation and actions on catalogs.

        Many of these parameters are taken care of by subclasses; this is the most general version.
        :param simname:
            Name of the cosmology. Generally same as the object.
        :param loc:
            Directory where halo catalogs are stored.
        :param filenames:
            Names of the files available/ of interest in loc. Interchangeable use-wise with filenames.
        :param cache_loc:
            The directory to cache the halocats for halotools.
        :param columns_to_keep:
            Which columns in halo catalog to keep. For more details, see halotools documentation.
        :param halo_finder:
            Which halo finder was used. For more details, see halotools documentation.
        :param version_name:
            The version name of the catalog. For more details, see halotools documentation.
        :param scale_factors:
            What scale factors are available/ of interest for this cosmology. Interchangeable use-wise with filenames.
        :param Lbox:
            Size of the box for this simulation.
        :param pmass:
            Mass of a single dark matter particle in the simulation.
        :param cosmo:
            An astropy.cosmology object with the cosmology of this simulation.
        '''

        # relevant information for Halotools
        self.simname = simname
        self.loc = loc
        self.columns_to_keep = columns_to_keep

        # TODO allow the user access to this? Probably.
        self.columns_to_convert = set(["halo_rvir", "halo_rs"])
        self.columns_to_convert = list(self.columns_to_convert & set(self.columns_to_keep.keys()))

        self.halo_finder = halo_finder
        self.version_name = version_name
        self.Lbox = Lbox
        self.pmass = pmass

        self.scale_factors = sorted(scale_factors)
        self.redshifts = [1.0 / a - 1 for a in
                          self.scale_factors]  # TODO let user pass in redshift and get a scale factor

        self.cosmology = cosmo  # default cosmology
        self.h = self.cosmology.H(0).value / 100.0

        # TODO Well this makes me think loc doesn't do anything...
        # I use it in the subclasses though. Doesn't mean I need it here though.
        # This one doesn't need to be easy to use; just general.
        self.filenames = sorted(filenames)
        for i, fname in enumerate(self.filenames):
            self.filenames[i] = self.loc + fname

        # Confirm filenames and scale_factors have the same length
        # halotools builtins have no filenames, so there is a case filenames = []
        assert (len(self.filenames) == len(self.redshifts)) or len(self.filenames) == 0

        self.cache_filenames = [path.join(cache_loc, 'hlist_%.2f.list.%s.hdf5' % (a, self.simname)) \
                                for a in self.scale_factors]

        self.halocat = None  # halotools halocat that we wrap
        self.model = None  # same as above, but for the model
        self.populated_once = False

    def __len__(self):
        '''Number of separate catalogs contained in one object. '''
        return len(self.scale_factors)

    def __str__(self):
        '''Return an informative output string.'''
        # TODO Some things could be removed, others added
        output = []
        output.append(self.simname)
        output.append('-' * 25)
        output.append('Halo finder:\t %s' % self.halo_finder)
        output.append('Version name:\t%s' % self.version_name)
        output.append('Cosmology:\n%s' % self.cosmology)
        output.append('Redshifts:\t%s' % str(self.redshifts))
        output.append('-' * 25)
        output.append('Location:\t%s' % self.loc)
        output.append('Lbox:\t%.1f' % self.Lbox)
        output.append('Particle Mass:\t%f' % self.pmass)
        output.append('Columns to Keep:\n%s' % str(self.columns_to_keep))

        return '\n'.join(output)

    def _update_lists(self, user_kwargs, tmp_fnames, tmp_scale_factors):
        '''If the user passes in a scale factor or filename, we have to do some cropping to ensure they align.
        Used by subclasses during initialization.'''

        if 'filenames' not in user_kwargs:
            user_kwargs['filenames'] = tmp_fnames
        elif 'scale_factors' in user_kwargs:  # don't know why this case would ever be true
            assert len(user_kwargs['filenames']) == len(user_kwargs['scale_factors'])
            for kw_fname in user_kwargs['filenames']:
                assert kw_fname in tmp_fnames
                # do nothing, we're good.
        else:
            user_kwargs['scale_factors'] = []
            for kw_fname in user_kwargs['filenames']:
                assert kw_fname in tmp_fnames
                user_kwargs['scale_factors'].append(
                    tmp_scale_factors[tmp_fnames.index(kw_fname)])  # get teh matching scale factor

        if 'scale_factors' not in user_kwargs:
            user_kwargs['scale_factors'] = tmp_scale_factors
        else:  # both case covered above.
            user_kwargs['filenames'] = []
            for a in user_kwargs['scale_factors']:
                assert a in tmp_scale_factors
                user_kwargs['filenames'].append(tmp_fnames[tmp_scale_factors.index(a)])  # get teh matching scale factor

    def _return_nearest_sf(self, a, tol=0.05):
        '''
        Find the nearest scale factor to a that is stored, within tolerance. If none exists, return None.
        :param a:
            Scale factor to find nearest neighbor to.
        :param tol:
            Tolerance within which "near" is defined. Default is 0.05
        :return: If a nearest scale factor is found, returns it. Else, returns None.
        '''
        assert 0 < a < 1  # assert a valid scale factor
        if a in self.scale_factors:  # try for an exact match.
            return a
        idx = np.argmin(np.abs(np.array(self.scale_factors) - a))
        if np.abs(self.scale_factors[idx] - a) < tol:
            return self.scale_factors[idx]
        else:
            return None

    def _check_cores(self, n_cores ):
        '''
        Helper function that checks that a user's input for the number of cores is sensible,
        returns modifications and issues warnings as necessary.
        :param n_cores:
            User input for number of cores
        :return:
            n_cores: A sensible number of cores given context and requirements.
        '''

        assert n_cores == 'all' or n_cores > 0
        if type(n_cores) is not str:
            assert int(n_cores) == n_cores

        max_cores = cpu_count()
        if n_cores == 'all':
            return max_cores
        elif n_cores > max_cores:
            warnings.warn('n_cores invalid. Changing from %d to maximum %d.' % (n_cores, max_cores))
            return max_cores
            # else, we're good!
        else:
            return n_cores

    def cache(self, scale_factors='all', overwrite=False):
        '''
        Cache a halo catalog in the halotools format, for later use.
        :param scale_factors:
            Scale factors to cache. Default is 'all', which will cache ever scale_factor stored.
        :param overwrite:
            Overwrite the file currently on disk, if it exists. Default is false.
        :return: None
        '''
        for a, z, fname, cache_fnames in izip(self.scale_factors, self.redshifts, self.filenames, self.cache_filenames):
            # TODO get right reader for each halofinder.
            if scale_factors != 'all' and a not in scale_factors:
                continue
            reader = RockstarHlistReader(fname, self.columns_to_keep, cache_fnames, self.simname,
                                         self.halo_finder, z, self.version_name, self.Lbox, self.pmass,
                                         overwrite=overwrite)
            reader.read_halocat(self.columns_to_convert, write_to_disk=True)

    def load(self, scale_factor, HOD='redMagic', tol=0.05):
        '''
        Load both a halocat and a model to prepare for population and calculation.
        :param scale_factor:
            Scale factor of the catalog to load. If no exact match found, searches for nearest in tolerance.
        :param HOD:
            HOD model to load. Currently available options are redMagic, stepFunc, and the halotools defatuls.
        :return: None
        '''
        a = self._return_nearest_sf(scale_factor, tol)
        if a is None:
            raise ValueError('Scale factor %.3f not within given tolerance.' % scale_factor)
        self.load_catalog(a, tol, check_sf=False)
        self.load_model(a, HOD, check_sf=False)

    def load_catalog(self, scale_factor, tol=0.05, check_sf=True):
        '''
        Load only a specific catalog. Not reccomended to use separately from broader load function.
        Its possible for the redshift ni the model to be different fro the one from the catalog,
        though the difference will be small.
        :param a:
            The scale factor of the catalog of interest
        :param check_sf:
            Boolean whether or not to use the passed in scale_factor blindly. Default is false.
        :return: None
        '''
        if check_sf:
            a = self._return_nearest_sf(scale_factor, tol)
            if a is None:
                raise ValueError('Scale factor %.3f not within given tolerance.' % scale_factor)
        else:
            a = scale_factor#YOLO
        z = 1.0/a-1

        self.halocat = CachedHaloCatalog(simname=self.simname, halo_finder=self.halo_finder,
                                         version_name=self.version_name, redshift=z)

    def load_model(self, scale_factor, HOD='redMagic', check_sf=True):
        '''
        Load an HOD model. Not reccomended to be used separately from the load function. It
        is possible for the scale_factor of the model and catalog to be different.
        :param scale_factor:
            Scale factor for the model
        :param HOD:
            HOD model to load. Currently available options are redMagic, stepFunc, and the halotools defatuls.
        :param check_sf:
            Boolean whether or not to use the passed in scale_factor blindly. Default is false.
        :return: None
        '''

        if check_sf:
            a = self._return_nearest_sf(scale_factor, tol)
            if a is None:
                raise ValueError('Scale factor %.3f not within given tolerance.' % scale_factor)
        else:
            a = scale_factor#YOLO
        z = 1.0/a-1

        assert HOD in {'redMagic', 'stepFunc', 'zheng07', 'leauthaud11', 'tinker13', 'hearin15'}

        if HOD == 'redMagic':
            cens_occ = RedMagicCens(redshift=z)
            sats_occ = RedMagicSats(redshift=z, cenocc_model=cens_occ)

            self.model = HodModelFactory(
                centrals_occupation=cens_occ,
                centrals_profile=TrivialPhaseSpace(redshift=z),
                satellites_occupation=sats_occ,
                satellites_profile=NFWPhaseSpace(redshift=z))

        elif HOD == 'stepFunc':
            cens_occ = StepFuncCens(redshift=z)
            sats_occ = StepFuncSats(redshift=z)

            self.model = HodModelFactory(
                centrals_occupation=cens_occ,
                centrals_profile=TrivialPhaseSpace(redshift=z),
                satellites_occupation=sats_occ,
                satellites_profile=NFWPhaseSpace(redshift=z))

        else:
            self.model = PrebuiltHodModelFactory(HOD)


    def populate(self, params={}, min_ptcl=200):
        '''
        Populate the stored halocatalog with a new realization. Load must be called first.
        :param params:
            HOD parameters. Only those that are changed from the original are required; the rest will remain the default.
        :param min_ptcl:
            Minimum number of particles which constitutes a halo.
        :return: None
        '''
        try:
            assert self.model is not None
        except AssertionError:
            raise AssertionError("Please call load before calling populate.")

        self.model.param_dict.update(params)
        # might be able to check is model has_attr mock.
        if self.populated_once:
            self.model.mock.populate(Num_ptcl_requirement=min_ptcl)
        else:
            self.model.populate_mock(self.halocat, Num_ptcl_requirement=min_ptcl)
            self.populated_once = True

    @observable
    def calc_number_density(self):
        '''
        Return the number density for a populated box.
        :return: Number density of a populated box.
        '''
        return len(self.model.mock.galaxy_table['x']) / (self.Lbox ** 3)

    @observable
    def calc_xi(self, rbins, n_cores='all', do_jackknife=True, use_corrfunc=False):
        '''
        Calculate a 3-D correlation function on a populated catalog.
        :param rbins:
            Radial bins for the correlation function.
        :param do_jackknife:
            Whether or not to do a jackknife along with the xi calculation. Generally slower. Not supported with
            with corrfunc at present.
        :param use_corrfunc:
            Whether or not to use the halotools function to calculate xi, or Manodeep's corrfunc. Corrfunc is not
            available on all systems and currently jackknife is not supported.
        :return: xi:
                len(rbins) 3-D correlation function
                xi_cov (if do_jacknife and ! use_corrfunc):
                (len(rbins), len(rbins)) covariance matrix for xi.
        '''
        assert do_jackknife != use_corrfunc  # xor
        if use_corrfunc and not CORRFUNC_AVAILABLE:
            # NOTE just switch to halotools in this case? Or fail?
            raise ImportError("Corrfunc is not available on this machine!")

        n_cores = self._check_cores(n_cores)

        x, y, z = [self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]
        pos = return_xyz_formatted_array(x, y, z, period=self.Lbox)

        if use_corrfunc:
            # write bins to file
            # unforunately how corrfunc has to work
            # TODO A custom binfile, or one that's already written?
            bindir = path.dirname(path.abspath(__file__))  # location of files with bin edges
            with open(path.join(bindir, './binfile'), 'w') as f:
                for low, high in zip(rbins[:-1], rbins[1:]):
                    f.write('\t%f\t%f\n' % (low, high))

            # countpairs requires casting in order to work right.
            xi_all = countpairs_xi(self.model.mock.Lbox * self.h, n_cores, path.join(bindir, './binfile'),
                                   x.astype('float32') * self.h, y.astype('float32') * self.h,
                                   z.astype('float32') * self.h)
            xi_all = np.array(xi_all, dtype='float64')[:, 3]

        else:
            if do_jackknife:
                np.random.seed(int(time()))
                # TODO customize these?
                n_rands = 5
                n_sub = 5

                randoms = np.random.random((pos.shape[0] * n_rands,
                                            3)) * self.Lbox * self.h  # Solution to NaNs: Just fuck me up with randoms
                xi_all, xi_cov = tpcf_jackknife(pos * self.h, randoms, rbins, period=self.Lbox * self.h,
                                                num_threads=n_cores, Nsub=n_sub)
            else:
                xi_all = tpcf(pos * self.h, rbins, period=self.Lbox * self.h, num_threads=n_cores())

        # TODO 1, 2 halo terms?

        if do_jackknife:
            return xi_all, xi_cov
        return xi_all

    # TODO use Joe's code. Remember to add sensible asserts when I do.
    # TODO Jackknife? A way to do it without Joe's code?
    @observable
    def calc_wp(self,rp_bins,pi_max=40, n_cores='all',RSD=True):
        '''
        Calculate the projected correlation on a populated catalog
        :param rp_bins:
            Radial bins of rp, distance perpendicular to projection direction.
        :param pi_max:
            Maximum distance to integrate to in projection. Default is 40 Mpc/h
        :param n_cores:
            Number of cores to use for calculation. Default is 'all' to use all available.
        :param RSD:
            Boolean whether or not to apply redshift space distortions. Default is True.
        :return:
            wp_all, the projection correlation function.
        '''

        n_cores = self._check_cores(n_cores)

        x, y, z = [self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]
        if RSD:
            # for now, hardcode 'z' as the distortion dimension.
            distortion_dim = 'z'
            v_distortion_dim = self.model.mock_table['v%s'%distortion_dim]
            #apply redshift space distortions
            pos = return_xyz_formatted_array(x,y,z,velocity=v_distortion_dim, \
                                velocity_distortion_dimension=distortion_dim, period = self.Lbox)
        else:
            pos = return_xyz_formatted_array(x, y, z, period=self.Lbox)

        #don't forget little h!!
        wp_all = wp(pos*self.h, rp_bins, pi_max*self.h, period=self.Lbox*self.h, num_threads=n_cores)
        return wp_all

    @observable
    def calc_wt(self,theta_bins, n_cores='all'):
        '''
        Calculate the angular correlation function, w(theta), from a populated catalog.
        :param theta_bins:
            The bins in theta to use.
        :param n_cores:
            Number of cores to use for calculation. Default is 'all' to use all available.
        :return:
            wt_all, the angular correlation function.
        '''

        n_cores = self._check_cores(n_cores)

        pos = np.vstack([self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]).T
        vels = np.vstack([self.model.mock.galaxy_table[c] for c in ['vx', 'vy', 'vz']]).T

        # TODO is the model cosmo same as the one attached to the cat?
        ra, dec, z = mock_survey.ra_dec_z(pos*self.h, vels*self.h, cosmo=self.model.mock.cosmology)
        ang_pos = np.vstack((np.degrees(ra), np.degrees(dec) )).T

        n_rands=5
        rand_pos = np.random.random((pos.shape[0] * n_rands,3)) * self.Lbox * self.h
        rand_vels = np.zeros((pos.shape[0]*n_rands, 3))

        rand_ra, rand_dec, rand_z = mock_survey.ra_dec_z(rand_pos*self.h, rand_vels*self.h, cosmo=self.model.mock.cosmology)
        rand_ang_pos = np.vstack((np.degree(rand_ra), np.degree(rand_dec))).T

        #NOTE I can transform coordinates and not have to use randoms at all. Consider?
        wt_all = angular_tpcf(ang_pos, theta_bins, randoms=rand_ang_pos, num_threads=n_cores)
        return wt_all

