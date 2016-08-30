#!bin/bash
#TODO better docstring
'''Module containing the main Cat object, which implements caching, loading, populating, and calculating observables
on catalogs. '''

from os import path
from itertools import izip
import numpy as np
from astropy import cosmology
from halotools.sim_manager import RockstarHlistReader, CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.empirical_models import HodModelFactory, TrivialPhaseSpace, NFWPhaseSpace
from customHODModels import *


class Cat(object):
    def __init__(self, simname='cat', loc='',filenames=[],cache_loc='/u/ki/swmclau2/des/halocats/',
                 columns_to_keep={}, halo_finder='rockstar',version_name='most_recent',
                 Lbox=1.0, pmass=1.0,scale_factors=[],cosmo=cosmology.WMAP5,
                **kwargs):

        #TODO need both loc and filenames?
        #TODO emply filenames does a glob?
        #TODO change 'loc' to dir?
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

        #relevant information for Halotools
        self.simname = simname
        self.loc = loc
        self.columns_to_keep = columns_to_keep

        #TODO allow the user access to this? Probably.
        self.columns_to_convert = set(["halo_rvir", "halo_rs"])
        self.columns_to_convert = list(self.columns_to_convert & set(self.columns_to_keep.keys()))

        self.halo_finder = halo_finder
        self.version_name = version_name
        self.Lbox = Lbox
        self.pmass = pmass

        self.scale_factors = scale_factors
        self.redshifts = [1.0 / a - 1 for a in
                          self.scale_factors]  # TODO let user pass in redshift and get a scale factor

        self.cosmology = cosmo  # default cosmology
        self.h = self.cosmology.H(0).value / 100.0

        #TODO Well this makes me think loc doesn't do anything...
        #I use it in the subclasses though. Doesn't mean I need it here though.
        #This one doesn't need to be easy to use; just general.
        self.filenames = filenames
        for i, fname in enumerate(self.filenames):
            self.filenames[i] = self.loc + fname

        # Confirm filenames and scale_factors have the same length
        # halotools builtins have no filenames, so there is a case filenames = []
        assert (len(self.filenames) == len(self.redshifts)) or len(self.filenames) == 0

        self.cache_filenames = [path.join(cache_loc, 'hlist_%.2f.list.%s.hdf5'%(a,self.simname))\
                                                                    for a in self.scale_factors]

        self.halocat = None #halotools halocat that we wrap
        self.model = None #same as above, but for the model
        self.populated_once = False

    def __len__(self):
        '''Number of separate catalogs contained in one object. '''
        return len(self.scale_factors)

    def __str__(self):
        '''Return an informative output string.'''
        #TODO Some things could be removed, others added
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

    def _return_nearest_sf(self, a, tol = 0.05):
        '''
        Find the nearest scale factor to a that is stored, within tolerance. If none exists, return None.
        :param a:
            Scale factor to find nearest neighbor to.
        :param tol:
            Tolerance within which "near" is defined. Default is 0.05
        :return: If a nearest scale factor is found, returns it. Else, returns None.
        '''
        assert 0<a<1 #assert a valid scale factor
        if a in self.scale_factors:# try for an exact match.
            return a
        idx = np.argmin(np.abs(np.array(self.scale_factors) - a))
        if np.abs(self.scale_factors[idx] - a) < tol:
            return self.scale_factors[idx]
        else:
            return None

    def cache(self, scale_factors = 'all',overwrite=False):
        '''
        Cache a halo catalog in the halotools format, for later use.
        :param scale_factors:
            Scale factors to cache. Default is 'all', which will cache ever scale_factor stored.
        :param overwrite:
            Overwrite the file currently on disk, if it exists. Default is false.
        :return: None
        '''
        for a,z, fname, cache_fnames in izip(self.scale_factors, self.redshifts, self.filenames, self.cache_filenames):
            #TODO get right reader for each halofinder.
            if scale_factors !='all' and a not in scale_factors:
                continue
            reader = RockstarHlistReader(fname, self.columns_to_keep, cache_fnames, self.simname,
                                         self.halo_finder, z, self.version_name, self.Lbox, self.pmass, overwrite=overwrite)
            reader.read_halocat(self.columns_to_convert, write_to_disk=True, overwrite=overwrite)

    #NOTE separate functions for loading model or halocat?
    def load(self, scale_factor, HOD='redMagic', tol = 0.05):
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
            raise ValueError('Scale factor %.3f not within given tolerance.'%scale_factor)
        z = 1/(1+a)

        assert HOD in {'redMagic', 'stepFunc', 'zheng07', 'leauthaud11', 'tinker13', 'hearin15'}

        if HOD=='redMagic':
            cens_occ = RedMagicCens(redshift=z)
            sats_occ = RedMagicSats(redshift=z, cenocc_model=cens_occ)

            self.model = HodModelFactory(
            centrals_occupation=cens_occ,
            centrals_profile=TrivialPhaseSpace(redshift=z),
            satellites_occupation=sats_occ,
            satellites_profile=NFWPhaseSpace(redshift=z))

        elif HOD=='stepFunc':
            cens_occ = StepFuncCens(redshift=z)
            sats_occ = StepFuncSats(redshift=z)

            self.model = HodModelFactory(
            centrals_occupation=cens_occ,
            centrals_profile=TrivialPhaseSpace(redshift=z),
            satellites_occupation=sats_occ,
            satellites_profile=NFWPhaseSpace(redshift=z))

        else:
            self.model=PrebuiltHodModelFactory(HOD)

        self.halocat = CachedHaloCatalog(simname = self.simname, halo_finder=self.halo_finder,
                                         version_name=self.version_name, redshift= z)

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
        #might be able to check is model has_attr mock.
        if self.populated_once:
            self.model.mock.populate_mock(self.halocat, Num_ptcl_requirement=min_ptcl)
        else:
            self.model.populate_mock(self.halocat, Num_ptcl_requirement=min_ptcl)
            self.populated_once = True

    def calc_number_density(self):



