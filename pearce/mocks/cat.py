#!bin/bash
#TODO better docstring
'''Module containing the main Cat object, which implements caching, loading, populating, and calculating observables
on catalogs. '''

from os import path
from itertools import izip
from astropy import cosmology
from halotools.sim_manager import RockstarHlistReader

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
        self.modle = None #same as above, but for the model

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

    def cache(self, scale_factors = 'all'):
        '''Use halotools cache to store a halocat for future use. Needs to be done before any other action'''
        for a,z, fname, cache_fnames in izip(self.scale_factors, self.redshifts, self.filenames, self.cache_filenames):
            #TODO get right reader for each halofinder.
            reader = RockstarHlistReader
    def load(self, scale_factor, HOD='redMagic'):
        '''Load a cached halocat and an HOD model'''
