#!/bin/bash
# TODO better docstring
'''Module containing the main Cat object, which implements caching, loading, populating, and calculating observables
on catalogs. '''

from os import path
from itertools import izip
from multiprocessing import cpu_count
import warnings
import inspect
from time import time
from glob import glob

from astropy import cosmology
from halotools.sim_manager import RockstarHlistReader, CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.empirical_models import HodModelFactory, TrivialPhaseSpace, NFWPhaseSpace
from halotools.mock_observables import *  # i'm importing so much this is just easier

from .customHODModels import *

# try to import corrfunc, determine if it was successful
try:
    from Corrfunc.theory import xi, wp

    CORRFUNC_AVAILABLE = True
except ImportError:
    CORRFUNC_AVAILABLE = False


def observable(func):
    '''
    Decorator for observable methods. Checks that the catalog is properly loaded and calcualted.
    :param func:
        Function that calculates an observable on a populated catalog.
    :return: func
    '''

    def _func(self, *args, **kwargs):

        if 'use_corrfunc' in kwargs and (kwargs['use_corrfunc'] and not CORRFUNC_AVAILABLE):
            # NOTE just switch to halotools in this case? Or fail?
            raise ImportError("Corrfunc is not available on this machine!")

        try:
            assert self.halocat is not None
            assert self.model is not None
            assert self.populated_once
        except AssertionError:
            raise AssertionError("The cat must have a loaded model and catalog and be populated before calculating an observable.")
        return func(self, *args, **kwargs)

    # store the arguments, as the decorator destroys the spec
    _func.args = inspect.getargspec(func)[0]

    return _func


class Cat(object):
    def __init__(self, simname='cat', loc='', filenames=[], cache_loc='/u/ki/swmclau2/des/halocats/',
                 columns_to_keep={}, halo_finder='rockstar', version_name='most_recent',
                 Lbox=1.0, pmass=1.0, scale_factors=[], cosmo=cosmology.WMAP5, gadget_loc='',
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
        self.columns_to_convert = set(["halo_rvir", "halo_rs","halo_rs_klypin"])
        self.columns_to_convert = list(self.columns_to_convert & set(self.columns_to_keep.keys()))

        self.halo_finder = halo_finder
        self.version_name = version_name
        self.Lbox = Lbox
        self.pmass = pmass

        self.scale_factors = np.array(sorted(scale_factors))
        self.redshifts = [1.0 / a - 1 for a in
                          self.scale_factors]  # TODO let user pass in redshift and get a scale factor

        self.cosmology = cosmo  # default cosmology
        self.h = self.cosmology.H(0).value / 100.0

        self.gadget_loc = gadget_loc  # location of original gadget files.

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

        sf_idxs = []  # store the indicies, as they have applications elsewhere

        if 'filenames' in user_kwargs and 'scale_factors' in user_kwargs:
            assert len(user_kwargs['filenames']) == len(user_kwargs['scale_factors'])
            for kw_fname in user_kwargs['filenames']:
                # Store indicies. If not in, will throw and error.
                sf_idxs.append(tmp_fnames.index(kw_fname))
                # do nothing, we're good.
        elif 'scale_factors' in user_kwargs:
            user_kwargs['filenames'] = []
            for a in user_kwargs['scale_factors']:
                idx = tmp_scale_factors.index(a)  # will raise an error if it's not there
                sf_idxs.append(idx)
                user_kwargs['filenames'].append(tmp_fnames[idx])  # get teh matching scale factor
        elif 'filenames' in user_kwargs:
            user_kwargs['scale_factors'] = []
            for kw_fname in user_kwargs['filenames']:
                idx = tmp_fnames.index(kw_fname)  # will throw an error if not in there.
                sf_idxs.append(idx)
                user_kwargs['scale_factors'].append(tmp_scale_factors[idx])  # get teh matching scale factor
        else:
            user_kwargs['filenames'] = tmp_fnames
            user_kwargs['scale_factors'] = tmp_scale_factors
            sf_idxs = range(len(tmp_scale_factors))

        self.sf_idxs = np.array(sf_idxs)



    def _return_nearest_sf(self, a, tol=0.05):
        '''
        Find the nearest scale factor to a that is stored, within tolerance. If none exists, return None.
        :param a:
            Scale factor to find nearest neighbor to.
        :param tol:
            Tolerance within which "near" is defined. Default is 0.05
        :return: If a nearest scale factor is found, returns it. Else, returns None.
        '''
        assert 0 < a <= 1  # assert a valid scale factor
        if a in self.scale_factors:  # try for an exact match.
            return a
        idx = np.argmin(np.abs(np.array(self.scale_factors) - a))
        if np.abs(self.scale_factors[idx] - a) < tol:
            return self.scale_factors[idx]
        else:
            return None

    def _check_cores(self, n_cores):
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

    def cache(self, scale_factors='all', overwrite=False, add_local_density=False):
        '''
        Cache a halo catalog in the halotools format, for later use.
        :param scale_factors:
            Scale factors to cache. Default is 'all', which will cache ever scale_factor stored.
        :param overwrite:
            Overwrite the file currently on disk, if it exists. Default is false.
        :return: None
        '''
        try:
            assert not add_local_density or self.gadget_loc  # gadget loc must be nonzero here!
        except:
            raise AssertionError('Cannot add local density without gadget location for %s' % self.simname)

        if add_local_density:
            all_snapdirs = glob(path.join(self.gadget_loc, 'snapdir*'))
            snapdirs = [all_snapdirs[idx] for idx in self.sf_idxs]#only the snapdirs for the scale factors were interested in .
        else:
            snapdirs = ['' for i in self.scale_factors]

        for a, z, fname, cache_fnames, snapdir in izip(self.scale_factors, self.redshifts, self.filenames, self.cache_filenames, snapdirs):
            # TODO get right reader for each halofinder.
            if scale_factors != 'all' and a not in scale_factors:
                continue
            reader = RockstarHlistReader(fname, self.columns_to_keep, cache_fnames, self.simname,
                                         self.halo_finder, z, self.version_name, self.Lbox, self.pmass,
                                         overwrite=overwrite)
            reader.read_halocat(self.columns_to_convert)
            if add_local_density:
                self.add_local_density(reader, snapdir)  # TODO how to add radius?

            reader.write_to_disk()  # do these after so we have a halo table to work off of
            reader.update_cache_log()

    def add_local_density(self, reader, snapdir, radius=[1, 5]):#[1,5,10]
        """
        Calculates the local density around each halo and adds it to the halo table, to be cached.
        :param reader:
            A RockstartHlistReader object from Halotools.
        :param snapdir:
            Gadget snapshot corresponding with this rockstar file.
        :param radius:
            Radius (in Mpc/h) around which to search for particles to estimate locas density. Default is 5 Mpc.
        :return: None
        """
        #doing imports here since these are both files i've stolen from Yao
        #Possible this will be slow
        from .readGadgetSnapshot import readGadgetSnapshot
        from fast3tree import fast3tree
        p = 1e-2
        all_particles = np.array([], dtype='float32')
        # TODO should fail gracefully if memory is exceeded or if p is too small.
        for file in glob(path.join(snapdir, 'snapshot*')):
            # TODO should find out which is "fast" axis and use that.
            # Numpy uses fortran ordering.
            particles = readGadgetSnapshot(file, read_pos=True)[1]  # Think this returns some type of tuple; should check
            downsample_idxs = np.random.choice(particles.shape[0], size =int( particles.shape[0]*p))
            particles = particles[downsample_idxs, :]
            #particles = particles[np.random.rand(particles.shape[0]) < p]  # downsample
            if particles.shape[0] == 0:
                continue

            all_particles = np.resize(all_particles, (all_particles.shape[0] + particles.shape[0], 3))
            all_particles[-particles.shape[0]:, :] = particles

        if type(radius) == float:
            radius = np.array([radius])
        elif type(radius) == list:
            radius = np.array(radius)
        densities = np.ones((reader.halo_table['halo_x'].shape[0], radius.shape[0]))

        with fast3tree(all_particles) as tree:
            for r_idx, r in enumerate(radius):
                densities[:, r_idx] = densities[:, r_idx]/ (p * 4 * np.pi / 3 * r ** 3)
                for idx, halo_pos in enumerate(
                        izip(reader.halo_table['halo_x'], reader.halo_table['halo_y'], reader.halo_table['halo_z'])):
                    particle_idxs = tree.query_radius(halo_pos, r, periodic=True)
                    densities[idx,r_idx] *= reader.particle_mass * len(particle_idxs)

            reader.halo_table['halo_local_density_%d'%(int(r))] = densities[:, r_idx]

    def load(self, scale_factor, HOD='redMagic', tol=0.05, hod_kwargs = {}):
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
        self.load_model(a, HOD, check_sf=False, hod_kwargs=hod_kwargs)

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
            a = scale_factor  # YOLO
        z = 1.0 / a - 1

        self.halocat = CachedHaloCatalog(simname=self.simname, halo_finder=self.halo_finder,
                                         version_name=self.version_name, redshift=z)

    # TODO not sure if assembias should be boolean, or keep it as separate HODs?
    def load_model(self, scale_factor, HOD='redMagic', check_sf=True, hod_kwargs={}):
        '''
        Load an HOD model. Not reccomended to be used separately from the load function. It
        is possible for the scale_factor of the model and catalog to be different.
        :param scale_factor:
            Scale factor for the model
        :param HOD:
            HOD model to load. Currently available options are redMagic, stepFunc, and the halotools defatuls.
            Also may pass in a tuple of cens and sats classes, which will be instantiated here.
        :param check_sf:
            Boolean whether or not to use the passed in scale_factor blindly. Default is false.
        :param hod_kwargs:
            Kwargs to pass into the HOD model being loaded. Default is none.
        :return: None
        '''

        if check_sf:
            a = self._return_nearest_sf(scale_factor)
            if a is None:
                raise ValueError('Scale factor %.3f not within given tolerance.' % scale_factor)
        else:
            a = scale_factor  # YOLO
        z = 1.0 / a - 1
        if type(HOD) is str:
            assert HOD in {'redMagic', 'hsabRedMagic','abRedMagic', 'reddick14','hsabReddick14','abReddick14','stepFunc',\
                           'zheng07', 'leauthaud11', 'tinker13', 'hearin15', 'reddick14+redMagic'}

            if HOD in {'redMagic', 'hsabRedMagic','abRedMagic', 'reddick14','stepFunc', 'reddick14+redMagic',\
                        'hsabReddick14','abReddick14'}: #my custom ones:
                if HOD == 'redMagic':
                    cens_occ = RedMagicCens(redshift=z, **hod_kwargs)
                    sats_occ = RedMagicSats(redshift=z, cenocc_model=cens_occ, **hod_kwargs)

                elif HOD == 'abRedMagic':
                    cens_occ = AssembiasRedMagicCens(redshift=z, **hod_kwargs)
                    sats_occ = AssembiasRedMagicSats(redshift=z, cenocc_model=cens_occ, **hod_kwargs)

                elif HOD == 'hsabRedMagic':
                    cens_occ = HSAssembiasRedMagicCens(redshift=z, **hod_kwargs)
                    sats_occ = HSAssembiasRedMagicSats(redshift=z, cenocc_model=cens_occ, **hod_kwargs)

                elif HOD == 'reddick14':
                    cens_occ = Reddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = Reddick14Sats(redshift=z, cenocc_model = cens_occ,**hod_kwargs) # no modulation
                elif HOD == 'hsabReddick14':
                    cens_occ = HSAssembiasReddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = HSAssembiasReddick14Sats(redshift=z, cenocc_model = cens_occ,**hod_kwargs) # no modulation
                elif HOD == 'abReddick14':
                    cens_occ = AssembiasReddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = AssembiasReddick14Sats(redshift=z, cenocc_model = cens_occ,**hod_kwargs) # no modulation
                elif HOD == 'stepFunc':
                    cens_occ = StepFuncCens(redshift=z, **hod_kwargs)
                    sats_occ = StepFuncSats(redshift=z, **hod_kwargs)
                # TODO make it so I can pass in custom HODs like this.
                # This will be obtuse when I include assembly bias
                elif HOD == 'reddick14+redMagic':
                    cens_occ = Reddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = RedMagicSats(redshift=z, cenocc_model = cens_occ, **hod_kwargs)

                self.model = HodModelFactory(
                    centrals_occupation=cens_occ,
                    centrals_profile=TrivialPhaseSpace(redshift=z),
                    satellites_occupation=sats_occ,
                    satellites_profile=NFWPhaseSpace(redshift=z))

            else:
                self.model = PrebuiltHodModelFactory(HOD, **hod_kwargs)
        else:
            cens_occ = HOD[0](redshift=z, **hod_kwargs)
            sats_occ = HOD[1](redshift=z, **hod_kwargs)
            self.model = HodModelFactory(
                centrals_occupation=cens_occ,
                centrals_profile=TrivialPhaseSpace(redshift=z),
                satellites_occupation=sats_occ,
                satellites_profile=NFWPhaseSpace(redshift=z))

    def get_assembias_key(self, gal_type):
        '''
        Helper function to get the key of the assembly bias strength keys, as they are obscure to access.
        :param gal_type:
            Galaxy type to get Assembias strength. Options are 'centrals' and 'satellites'.
        :return:
        '''
        assert gal_type in {'centrals', 'satellites'}
        return self.model.input_model_dictionary['%s_occupation' % gal_type]._get_assembias_param_dict_key(0)

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
    def calc_number_density(self, halo=False):
        '''
        Return the number density for a populated box.
        :param: halo
            Whether to calculate hte number density of halos instead of galaxies. Default is false.
        :return: Number density of a populated box.
        '''
        if halo:
            return len(self.model.mock.halo_table['halo_x']) / (self.Lbox ** 3)
        #return len(self.model.mock.galaxy_table['x']) / (self.Lbox ** 3)
        return self.model.mock.number_density

    # TODO do_jackknife to cov?
    @observable
    def calc_xi(self, rbins, n_cores='all', do_jackknife=True, use_corrfunc=False, jk_args={}, halo=False):
        '''
        Calculate a 3-D correlation function on a populated catalog.
        :param rbins:
            Radial bins for the correlation function.
        :param n_cores
            Number of cores to use. default is 'all'
        :param do_jackknife:
            Whether or not to do a jackknife along with the xi calculation. Generally slower. Not supported with
            with corrfunc at present.
        :param use_corrfunc:
            Whether or not to use the halotools function to calculate xi, or Manodeep's corrfunc. Corrfunc is not
            available on all systems and currently jackknife is not supported.
        :param jk_args
            Dictionary of arguements for the jackknife call.
        :param halo
            Whether to calculate the halo correlation instead of the galaxy correlation. Default is False.
        :return: xi:
                len(rbins) 3-D correlation function
                xi_cov (if do_jacknife and ! use_corrfunc):
                (len(rbins), len(rbins)) covariance matrix for xi.
        '''
        assert not (do_jackknife and use_corrfunc)  # can't both be true.

        n_cores = self._check_cores(n_cores)
        if halo:
            x,y,z = [self.model.mock.halo_table[c] for c in ['halo_x', 'halo_y', 'halo_z']]
        else:
            x, y, z = [self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]
        pos = return_xyz_formatted_array(x, y, z, period=self.Lbox)

        if use_corrfunc:
            '''
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
            '''
            out = xi(self.model.mock.Lbox * self.h, n_cores, rbins,
                     x.astype('float32') * self.h, y.astype('float32') * self.h,
                     z.astype('float32') * self.h)

            xi_all = out[4]  # returns a lot of irrelevant info
            # TODO jackknife with corrfunc?

        else:
            if do_jackknife:
                np.random.seed(int(time()))
                if not jk_args:
                    # TODO customize these?
                    n_rands = 5
                    n_sub = 5
                else:
                    n_rands = jk_args['n_rands']
                    n_sub = jk_args['n_sub']

                randoms = np.random.random((pos.shape[0] * n_rands,
                                            3)) * self.Lbox * self.h  # Solution to NaNs: Just fuck me up with randoms
                xi_all, xi_cov = tpcf_jackknife(pos * self.h, randoms, rbins, period=self.Lbox * self.h,
                                                num_threads=n_cores, Nsub=n_sub, estimator='Landy-Szalay')
            else:
                xi_all = tpcf(pos * self.h, rbins, period=self.Lbox * self.h, num_threads=n_cores,
                              estimator='Landy-Szalay')

        # TODO 1, 2 halo terms?

        if do_jackknife:
            return xi_all, xi_cov
        return xi_all

    # TODO use Joe's code. Remember to add sensible asserts when I do.
    # TODO Jackknife? A way to do it without Joe's code?
    @observable
    def calc_wp(self, rp_bins, pi_max=40, do_jackknife=True, use_corrfunc=False, n_cores='all', RSD=True, halo=False):
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
        :param halo:
            Whether to calculate halo correlation or galaxy correlation. Default is False.
        :return:
            wp_all, the projection correlation function.
        '''

        # could move these last parts ot the decorator or a helper. Ah well.
        n_cores = self._check_cores(n_cores)

        if halo:
            x,y,z = [self.model.mock.halo_table[c] for c in ['halo_x', 'halo_y', 'halo_z']]
        else:
            x, y, z = [self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]

        #No RSD for halos
        if RSD and not halo:
            # for now, hardcode 'z' as the distortion dimension.
            distortion_dim = 'z'
            v_distortion_dim = self.model.mock.galaxy_table['v%s' % distortion_dim]
            # apply redshift space distortions
            pos = return_xyz_formatted_array(x, y, z, velocity=v_distortion_dim, \
                                             velocity_distortion_dimension=distortion_dim, period=self.Lbox)
        else:
            pos = return_xyz_formatted_array(x, y, z, period=self.Lbox)

        # don't forget little h!!
        if use_corrfunc:
            out = xi(self.model.mock.Lbox * self.h, pi_max * self.h, n_cores, rp_bins,
                     x.astype('float32') * self.h, y.astype('float32') * self.h,
                     z.astype('float32') * self.h)

            wp_all = out[4]  # returns a lot of irrelevant info
        else:
            wp_all = wp(pos * self.h, rp_bins, pi_max * self.h, period=self.Lbox * self.h, num_threads=n_cores)
        return wp_all

    @observable
    def calc_wt(self, theta_bins, do_jackknife=True, n_cores='all', halo = False):
        '''
        Calculate the angular correlation function, w(theta), from a populated catalog.
        :param theta_bins:
            The bins in theta to use.
        :param n_cores:
            Number of cores to use for calculation. Default is 'all' to use all available.
        :param halo:
            Whether or not to calculate clustering or halos or galaxies. Default is False.
        :return:
            wt_all, the angular correlation function.
        '''

        n_cores = self._check_cores(n_cores)

        if halo:
            pos = np.vstack( [self.model.mock.halo_table[c] for c in ['halo_x', 'halo_y', 'halo_z']]).T
            vels = np.vstack([self.model.mock.halo_table[c] for c in ['halo_vx', 'halo_vy', 'halo_vz']]).T
        else:
            pos = np.vstack([self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]).T
            vels = np.vstack([self.model.mock.galaxy_table[c] for c in ['vx', 'vy', 'vz']]).T

        # TODO is the model cosmo same as the one attached to the cat?
        ra, dec, z = mock_survey.ra_dec_z(pos * self.h, vels * self.h, cosmo=self.cosmology)
        ang_pos = np.vstack((np.degrees(ra), np.degrees(dec))).T

        n_rands = 5
        rand_pos = np.random.random((pos.shape[0] * n_rands, 3)) * self.Lbox * self.h
        rand_vels = np.zeros((pos.shape[0] * n_rands, 3))

        rand_ra, rand_dec, rand_z = mock_survey.ra_dec_z(rand_pos * self.h, rand_vels * self.h, cosmo=self.cosmology)
        rand_ang_pos = np.vstack((np.degrees(rand_ra), np.degrees(rand_dec))).T

        # NOTE I can transform coordinates and not have to use randoms at all. Consider?
        wt_all = angular_tpcf(ang_pos, theta_bins, randoms=rand_ang_pos, num_threads=n_cores)
        return wt_all
