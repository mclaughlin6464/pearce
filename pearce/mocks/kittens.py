#!/bin/bash
'''Module with a clever name that holds all the custom subclasses of  Cat. Each is essentially
a holder for information to simplify the process of making Cats. It takes care of a lot of file handling/
busy work.

Each takes two kwargs: "filenames" and "scale_factors", both lists which allow the user to restrict
which particular files are cached/loaded/etc. '''

from glob import glob
from itertools import izip, product
from os import path
import numpy as np
from astropy import cosmology
from astropy import units as u
import astropy.constants as const
import pandas as pd
import h5py 
from ast import literal_eval
from halotools.sim_manager import RockstarHlistReader, UserSuppliedHaloCatalog, UserSuppliedPtclCatalog
from halotools.empirical_models import HodModelFactory, TrivialPhaseSpace, NFWPhaseSpace, PrebuiltHodModelFactory
from halotools.utils import add_halo_hostid
from .cat import Cat, HOD_DICT, VALID_HODS, DEFAULT_HODS
import sys
from time import time

__all__ = ['Bolshoi', 'Multidark', 'Emu', 'Fox', 'MDHR', 'Chinchilla', 'Aardvark', 'Guppy', 'cat_dict']

# Different table schema.
HLIST_COLS = {'halo_id': (1, 'i8'), 'halo_upid': (6, 'i8'),
              'halo_x': (17, 'f4'), 'halo_y': (18, 'f4'), 'halo_z': (19, 'f4'),
              'halo_vx': (20, 'f4'), 'halo_vy': (21, 'f4'), 'halo_vz': (22, 'f4'),
              'halo_mvir': (10, 'f4'), 'halo_rvir': (11, 'f4'), 'halo_rs': (12, 'f4'),
              'halo_snapnum': (31, 'i8'), 'halo_vpeak': (57, 'f4'), 'halo_halfmass_scale': (58, 'f4'),
              'halo_rs_klypin': (34, 'f4'), 'halo_vmax': (16, 'f4'), 'halo_macc': (54, 'f4'), 'halo_vacc': (56, 'f4'), 'halo_m200b':(36, 'f4')}

OUTLIST_COLS = {'halo_id': (0, 'i8'), 'halo_upid': (36, 'i8'),
                'halo_x': (8, 'f4'), 'halo_y': (9, 'f4'), 'halo_z': (10, 'f4'),
                'halo_vx': (11, 'f4'), 'halo_vy': (12, 'f4'), 'halo_vz': (13, 'f4'),
                'halo_mvir': (2, 'f4'), 'halo_rvir': (5, 'f4'), 'halo_rs': (6, 'f4')}

OUTLIST_BGC2_COLS = {'halo_id': (0, 'i8'), 'halo_upid': (14, 'i8'),
                     'halo_x': (8, 'f4'), 'halo_y': (9, 'f4'), 'halo_z': (10, 'f4'),
                     'halo_vx': (11, 'f4'), 'halo_vy': (12, 'f4'), 'halo_vz': (13, 'f4'),
                     'halo_m200b': (2, 'f4'), 'halo_r200b': (5, 'f4'), 'halo_rs': (6, 'f4')}


# Previously, I had Lbox and Npart be required for all, even tho it isn't actually. I don't know why now...
# I've also removed the updating of the kwargs with the defaults.
# The user shouldn't be updating thee cosmology, simname, etc. The only thing they can change
# are the scale factors that are loaded/cached (or their filenames). The only other thing
# maybe worth changing is hlist_cols, but I find that unlikely.

class Aardvark(Cat):
    # Lbox technically required, but I don't even have access to anything besides 400. Ignore for now.
    def __init__(self, system='ki-ls', **kwargs):

        # set these defaults
        simname = 'aardvark'
        cosmo = cosmology.core.LambdaCDM(H0=100 * 0.73, Om0=0.23, Ode0=0.77, Ob0=0.047)
        pmass = 4.75619e+08
        Lbox = 400.0
        columns_to_keep = HLIST_COLS

        # some of the columns here are different
        columns_to_keep['halo_vpeak'] = (52, 'f4')
        columns_to_keep['halo_mpeak'] = (50, 'f4')
        try:
            del columns_to_keep['halo_halfmass_scale']
        except KeyError:
            pass

        locations = {'ki-ls': '/nfs/slac/g/ki/ki18/des/mbusha/simulations/Aardvark-2PS/Lb400/rockstar/hlists/'}
        locations['long'] = locations['ki-ls']

        assert system in locations
        loc = locations[system]

        tmp_fnames = sorted(glob(loc + 'hlist_*.list'))  # snag all the hlists
        tmp_fnames = [fname[len(loc):] for fname in tmp_fnames]  # just want the names in the dir
        tmp_scale_factors = [float(fname[6:-5]) for fname in tmp_fnames]  # pull out scale factors

        # If use specified scale factors or filenames, select only those.
        # Looked into a way to put this in the global init.
        # However, the amount of copy-pasting that would save would be minimal, it turns out.
        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)
        # Now fnames and scale factors are rolled into kwargs
        new_kwargs = kwargs.copy()
        # delete the keys that are fixed, if they exist
        for key in ['simname', 'loc', 'columns_to_keep', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        super(Aardvark, self).__init__(simname, loc, columns_to_keep=columns_to_keep, Lbox=Lbox, \
                                       pmass=pmass, cosmo=cosmo, **new_kwargs)


class Bolshoi(Cat):
    'Builtin catalog, so needs very little. Will never be manually cached!'

    def __init__(self):
        simname = 'bolshoi'
        version_name = 'halotools_alpha_version2'
        scale_factors = [1.0]

        super(Bolshoi, self).__init__(simname, version_name=version_name, scale_factors=scale_factors)


class Chinchilla(Cat):
    'The most complicated (and biggest) subclass'

    def __init__(self, Lbox, npart=2048, system='ki-ls', updated_hlists=False, **kwargs):
        # Lbox required, npart should be if Lbox is not 1050!
        simname = 'chinchilla'
        cosmo = cosmology.core.LambdaCDM(H0=100 * 0.7, Om0=0.286, Ode0=0.714)
        gadget_loc = ''

        if Lbox not in {125.0, 250.0, 400.0, 1050.0}:
            raise ValueError("Invalid boxsize %.1f for Chinchilla" % Lbox)

        if updated_hlists and (Lbox != 400 or npart != 2048 or (system != 'ki-ls' or system != 'long')):
            raise ValueError("The updated_hlists option is only available for Chinchilla Lb400-2048")

        # the different box sizes are in differenct locations, among other things.
        if Lbox == 1050:
            locations = {'ki-ls': '/u/ki/swmclau2/des/chinchilla1050/'}
            assert system in locations  # necessary?
            loc = locations[system]

            columns_to_keep = OUTLIST_COLS
            pmass = 3.34881e+10
            npart = 1400
            version_name = 'Lb1050-1400'

            tmp_scale_factors = [0.1429, 0.1667, 0.2, 0.25, 0.3333, 0.4, 0.5, 0.6667, 0.8, 1.0]
            tmp_fnames = ['out_%d.list' % i for i in xrange(10)]

        elif updated_hlists:
            locations = {'ki-ls': '~jderose/desims/BCCSims/c400-2048/rockstar/hlists_new/'}
            locations['long'] = locations['ki-ls']

            loc = locations[system]
            columns_to_keep = UPDATED_HLIST_COLS

            # may also work for 1050 too.
            pmass = 1.44390e+08 * ((Lbox / 125.0) ** 3) * ((1024.0 / npart) ** 3)

            # check that this combination of lbox and npart exists
            version_name = 'Lb%d-%d' % (int(Lbox), npart)

            # particles still in the other directory
            gadget_loc = '/nfs/slac/g/ki/ki21/cosmo/yymao/sham_test/resolution-test/'
            gadget_loc += '/output/'

            tmp_fnames = sorted(glob(loc + 'hlist_[0,1].*.list'))  # snag all the hlists
            tmp_fnames = [fname[len(loc):] for fname in tmp_fnames]  # just want the names in the dir
            tmp_scale_factors = [float(fname[6:-5]) for fname in tmp_fnames]  # pull out scale factors
        else:
            locations = {'ki-ls': '/nfs/slac/g/ki/ki21/cosmo/yymao/sham_test/resolution-test/',
                         'sherlock': '/scratch/users/swmclau2/hlists/Chinchilla/'}
            locations['long'] = locations['ki-ls']

            assert system in locations
            loc = locations[system]
            columns_to_keep = HLIST_COLS

            # may also work for 1050 too.
            pmass = 1.44390e+08 * ((Lbox / 125.0) ** 3) * ((1024.0 / npart) ** 3)

            valid_version_names = {'Lb125-1024', 'Lb125-2048', 'Lb250-1024', 'Lb250-128',
                                   'Lb250-196', 'Lb250-2048', 'Lb250-2560', 'Lb250-320',
                                   'Lb250-512', 'Lb250-768', 'Lb250-85', 'Lb400-1024',
                                   'Lb400-136', 'Lb400-2048', 'Lb400-210', 'Lb400-315',
                                   'Lb400-512', 'Lb400-768'}

            # check that this combination of lbox and npart exists
            version_name = 'Lb%d-%d' % (int(Lbox), npart)
            assert version_name in valid_version_names
            # add a subdirectory
            loc += 'c%d-%d/' % (int(Lbox), int(npart))

            gadget_loc = loc
            loc += '/rockstar/hlists/'
            gadget_loc += '/output/'

            tmp_fnames = sorted(glob(loc + 'hlist_*.list'))  # snag all the hlists
            tmp_fnames = [fname[len(loc):] for fname in tmp_fnames]  # just want the names in the dir
            tmp_scale_factors = [float(fname[6:-5]) for fname in tmp_fnames]  # pull out scale factors

        # if the user passed in stuff, have to check a bunch of things
        self.npart = npart
        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'version_name', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        super(Chinchilla, self).__init__(simname=simname, loc=loc, columns_to_keep=columns_to_keep,
                                         version_name=version_name, Lbox=Lbox,
                                         pmass=pmass, cosmo=cosmo, gadget_loc=gadget_loc, **new_kwargs)
        # Chinchillas also have to be cached differently.
        cache_locs = {'ki-ls': '/u/ki/swmclau2/des/halocats/',
                      'sherlock': '/scratch/users/swmclau2/halocats/'}
        fname = 'hlist_%.2f.list.%s_%s.hdf5'
        cache_locs['long'] = cache_locs['ki-ls']
        self.cache_loc = cache_locs[system]
        self.cache_filenames = [path.join(self.cache_loc, fname % (a, self.simname, self.version_name)
 )                          for a in self.scale_factors]  # make sure we don't have redunancies.
    def _get_cosmo_param_names_vals(self):
        # TODO docs
        names = ['Omega_m', 'Omega_b','h', 'sigma8', 'n_s']
        vals = [0.286, 0.047,self.h, 0.82, 0.96]
        return names, vals


# TODO consider a name change, as there could be a conflict with the emulator object
class Emu(Cat):

    def __init__(self, Lbox, system='ki-ls', **kwargs):

        simname = 'emu'

        # The emu naming convention is different than the others; two Emu boxes of the same number don't have the same
        # cosmology. This will need to be expanded later.
        if Lbox == 1050.0:
            locations = {'ki-ls': '/u/ki/swmclau2/des/emu/Box000/',
                         'sherlock': '/scratch/users/swmclau2/hlists/emu/Box000/'}
            assert system in locations
            loc = locations[system]
            cosmo = cosmology.core.wCDM(H0=63.36569, Om0=0.340573, Ode0=0.659427, w0=-0.816597)
            columns_to_keep = OUTLIST_COLS
            pmass = 3.9876e10

            tmp_scale_factors = [0.25, 0.333, 0.5, 0.540541, 0.588235, 0.645161, 0.714286, 0.8, 0.909091, 1.0]
            tmp_fnames = ['out_%d.list' % i for i in xrange(10)]

        if Lbox == 200.0:
            locations = {'ki-ls': '/scratch/PI/kipac/yymao/highres_emu/Box000/hlists/'}
            assert system in locations
            loc = locations[system]
            cosmo = cosmology.core.wCDM(H0=100 * 0.6616172, Om0=0.309483642394, Ode0=0.690516357606,
                                        w0=-0.8588491)
            columns_to_keep = HLIST_COLS
            pmass = 6.3994e8

            tmp_fnames = sorted(glob(loc + 'hlist_*.list'))  # snag all the hlists
            tmp_fnames = [fname[len(kwargs['loc']):] for fname in tmp_fnames]  # just want the names in the dir
            tmp_scale_factors = [float(fname[6:-5]) for fname in tmp_fnames]

        version_name = 'most_recent_%d' % int(Lbox)

        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)
        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'version_name', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        super(Emu, self).__init__(simname=simname, loc=loc, columns_to_keep=columns_to_keep,
                                  version_name=version_name, Lbox=Lbox, pmass=pmass,
                                  cosmo=cosmo, **new_kwargs)


class Fox(Cat):
    def __init__(self, system='ki-ls', **kwargs):
        simname = 'fox'
        columns_to_keep = HLIST_COLS
        Lbox = 400.0  # only one available
        pmass = 6.58298e8
        cosmo = cosmology.core.LambdaCDM(H0=100 * 0.6704346, Om0=0.318340, Ode0=0.681660)
        locations = {'ki-ls': '/nfs/slac/g/ki/ki23/des/BCCSims/Fox/Lb400/halos/rockstar/output/hlists/',
                     'sherlock': '/scratch/users/swmclau2/hlists/Fox/'}
        assert system in locations
        loc = locations[system]

        tmp_fnames = ['hlist_%d' % n for n in [46, 57, 73, 76, 79, 82, 86, 90, 95, 99]]
        tmp_scale_factors = [0.25, 0.333, 0.5, 0.540541, 0.588235, 0.645161, 0.714286, 0.8, 0.909091, 1.0]

        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'Lbox', 'pmass']:
            if key in new_kwargs:
                del new_kwargs[key]

        super(Fox, self).__init__(simname=simname, loc=loc, columns_to_keep=columns_to_keep, Lbox=Lbox,
                                  pmass=pmass, **new_kwargs)


class Guppy(Cat):
    def __init__(self, system='ki-ls', **kwargs):
        simname = 'guppy'
        columns_to_keep = OUTLIST_COLS
        Lbox = 1050.0  # only one available
        pmass = 3.45420e+10
        cosmo = cosmology.core.LambdaCDM(H0=100 * 0.6881, Om0=0.295, Ode0=0.705)

        locations = {'ki-ls': '/nfs/slac/g/ki/ki23/des/BCCSims/Fox/Lb400/halos/rockstar/output/hlists/',
                     'sherlock': '/scratch/users/swmclau2/hlists/Fox/'}
        assert system in locations
        loc = locations[system]

        tmp_fnames = ['out_%d.list' % i for i in xrange(10)]
        tmp_scale_factors = [0.1429, 0.1667, 0.2, 0.25, 0.3333, 0.4, 0.5, 0.6667, 0.8, 1.0]

        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        super(Fox, self).__init__(simname=simname, loc=loc, columns_to_keep=columns_to_keep, Lbox=Lbox,
                                  pmass=pmass, cosmo=cosmo, **kwargs)


class Multidark(Cat):
    'Builtin, so needs very little. Will never be cached!'

    def __init__(self, **kwargs):
        simname = 'multidark'
        version_name = 'halotools_alpha_version2'
        scale_factors = [2.0 / 3, 1.0]

        super(Multidark, self).__init__(simname, version_name=version_name, scale_factors=scale_factors)


# TODO put this under Multidark? not sure I want to mix builtins and mine.
class MDHR(Cat):
    def __init__(self, system='ki-ls', **kwargs):

        simname = 'multidark_highres'
        columns_to_keep = HLIST_COLS
        Lbox = 1e3
        pmass = 8.721e9

        # MDHR uses the default WMAP cosmology

        locations = {'ki-ls': '/nfs/slac/g/ki/ki20/cosmo/behroozi/MultiDark/hlists/',
                     'sherlock': '/scratch/users/swmclau2/hlists/MDHR/'}
        assert system in locations
        loc = locations[system]

        tmp_scale_factors = [0.25690, 0.34800, 0.49990, 0.53030, 0.65180, 0.71250, 0.80370, 0.91000, 1.00110]
        tmp_fnames = ['hlist_%.5f.list' % a for a in tmp_scale_factors]

        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'version_name', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        if system == 'sherlock':
            cache_loc = '/scratch/users/swmclau2/halocats/'
        else:
            cache_loc = '/u/ki/swmclau2/des/halocats/'
        super(MDHR, self).__init__(simname, loc=loc, columns_to_keep=columns_to_keep, Lbox=Lbox,
                                   pmass=pmass, cache_loc = cache_loc, **new_kwargs)


class TrainingBox(Cat):

    # lazily loadin the params from disk
    # may wanna hardcode them ot the object instead
    # also won't work for non-kils systems
    #cosmo_params = pd.DataFrame.from_csv('~swmclau2/des/LH_eigenspace_lnA_np7_n40_s556.dat',\
    #                                     sep = ' ', index_col = None)


    def __init__(self, boxno, system='ki-ls', **kwargs):

        assert 0 <= boxno < 40
        assert int(boxno) == boxno #under 40, is an int


        self.boxno = boxno

        simname = 'trainingbox'# not sure if something with Emu would work better, but wanna separate from Emu..
        columns_to_keep =  OUTLIST_BGC2_COLS.copy()
        # Add duplicate columns for mvir if allowed
        # TODO this sucks but halotools is tough here.
        del columns_to_keep['halo_m200b']
        columns_to_keep['halo_mvir'] = (2, 'f4')
        del columns_to_keep['halo_r200b']
        columns_to_keep['halo_rvir'] = (5, 'f4')

        Lbox = 1050.0  # Mpc/h
        self.npart = 1400
        # Need to make a way to combine all the params
        param_file = './cosmos.txt' # may need an abs path, can get the __file__ path

        cosmo_params = pd.read_csv(param_file, sep = ' ', index_col = 0)
        self.cosmo_params = cosmo_params.drop('ln10As', axis = 1) # replaced this with sigma8, remove redundancy

        cosmo = self._get_cosmo()
        pmass = 3.98769e10 * cosmo.Om0/\
                ((self.cosmo_params.iloc[0]['omch2']+self.cosmo_params.iloc[0]['ombh2'])*100*100/(self.cosmo_params.iloc[0]['H0']**2))

        self.prim_haloprop_key = 'halo_m200b'
        #locations = {'ki-ls': ['/u/ki/swmclau2/des/testbox_findparents/']}
        locations = {#'ki-ls': ['/u/ki/swmclau2/des/NewAemulusBoxes/Box0%02d/',
                     #           '/u/ki/swmclau2/des/NewAemulusBoxes/Box0%02d/'],
                     'ki-ls': ['/nfs/slac/g/ki/ki22/cosmo/beckermr/tinkers_emu/Box0%02d/',
                               '/nfs/slac/g/ki/ki23/des/beckermr/tinkers_emu/Box0%02d/',
                               '/nfs/slac/g/ki/ki18/des/swmclau2/NewAemulusBoxes/Box0%02d/'],
                     'sherlock': ['/home/users/swmclau2/scratch/NewTrainingBoxes/Box0%02d/',
                                  '/home/users/swmclau2/scratch/NewTrainingBoxes/Box0%02d/']}
                      #same place on sherlock
        locations['long'] = locations['ki-ls']
        assert system in locations
        #loc = locations[system][0]
        loc_list = locations[system]

        in_first_loc = set([1,2,3,4,10,11,12,13,14,15,16,17,18,19,25,30,31,32,33,34])

        if boxno in in_first_loc:
            loc = loc_list[0]%(boxno)
        elif boxno == 0 and system== 'ki-ls': # TODO all the boxes should be updated this way
            loc = loc_list[2]%(boxno)
        else:
            loc = loc_list[1]%(boxno)

        if boxno!= 0 or system != 'ki-ls':
            gadget_loc = loc + 'output/'
        else:
            gadget_loc = loc_list[0]%(boxno) + 'output/' 
        loc += 'halos/m200b/'

        #TODO why not using rs boxes???
        tmp_fnames = ['outbgc2_rs_%d.list' % i for i in xrange(10)]
        tmp_scale_factors = [0.25, 0.333, 0.5, 0.540541, 0.588235, 0.645161, 0.714286, 0.8, 0.909091, 1.0]
        #tmp_scale_factors = [0.645161]
        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        version_name = 'most_recent_%02d'%boxno

        super(TrainingBox, self).__init__(simname=simname, loc=loc, columns_to_keep=columns_to_keep, Lbox=Lbox,
                                      pmass=pmass, version_name = version_name, cosmo=cosmo,gadget_loc=gadget_loc, **new_kwargs)

        cache_locs = {'ki-ls': '/u/ki/swmclau2/des/halocats/hlist_%.2f.list.%s_%02d.hdf5',
                      'sherlock': '/scratch/users/swmclau2/halocats/hlist_%.2f.list.%s_%02d.hdf5'}
        cache_locs['long'] = path.dirname(cache_locs['ki-ls'])
        self.cache_loc = path.dirname(cache_locs[system])#%(a, self.simname, boxno)
        self.cache_filenames = [cache_locs[system] % (a, self.simname, boxno)
                                for a in self.scale_factors]  # make sure we don't have redunancies.


    def _get_cosmo(self):
        """
        Construct the cosmology object taht corresponds to this boxnumber
        :param boxno:
            The box number whose cosmology we want
        :return:
            Cosmology, the FlatwCDM astropy cosmology with the parameters of boxno
        """
        params = self.cosmo_params.iloc[self.boxno]
        h = params['H0']/100.0
        Om0 = (params['ombh2'] + params['omch2'])/(h**2)
        return cosmology.core.FlatwCDM(H0= params['H0'], Om0 = Om0, Neff=params['Neff'], Ob0=params['ombh2']/(h**2),
                                       w0 = params['w0'])

    def _get_cosmo_param_names_vals(self):
        # TODO docs
        names = list(self.cosmo_params.columns.values)
        vals = np.array(list(self.cosmo_params.iloc[self.boxno].values))
        return names, vals


class TestBox(Cat):

    def __init__(self, boxno, realization, system='ki-ls', **kwargs):

        #       I'm writing a hack to work for the CMASS catalogs I'm making. I have to do something better in the future.
        # The hack is due to the bgc2 cats not having concetrations, and not being possible to match againts the outlists.
        # I ran findparents on 3 of the catalogs. ALl other catalogs will be in accessible for now
        #assert realization == 0
        #assert 0 <= boxno <= 2
        assert 0<=boxno<=6
        assert 0<=realization<=4

        self.boxno = boxno
        self.realization = realization

        simname = 'testbox'# not sure if something with Emu would work better, but wanna separate from Emu..
        columns_to_keep =  OUTLIST_BGC2_COLS.copy()
        # Add duplicate columns for mvir if allowed
        # TODO this sucks but halotools is tough here.
        del columns_to_keep['halo_m200b']
        columns_to_keep['halo_mvir'] = (2, 'f4')
        del columns_to_keep['halo_r200b']
        columns_to_keep['halo_rvir'] = (5, 'f4')

        Lbox = 1050.0  # Mpc/h
        self.npart = 1400
        # Need to make a way to combine all the params
        param_file = './test_cosmos.txt'  # may need an abs path, can get the __file__ path
        cosmo_params = pd.read_csv(param_file, sep=' ', index_col=0)
        self.cosmo_params = cosmo_params.drop('ln10As', axis=1)  # replaced this with sigma8, remove redundancy

        cosmo = self._get_cosmo()
        pmass =  3.83914e10* cosmo.Om0/\
                ((self.cosmo_params.iloc[0]['omch2']+self.cosmo_params.iloc[0]['ombh2'])*100*100/(self.cosmo_params.iloc[0]['H0']**2))

        self.prim_haloprop_key = 'halo_m200b'
        #locations = {'ki-ls': ['/u/ki/swmclau2/des/testbox_findparents/']}
        locations = {'ki-ls': ['/nfs/slac/des/fs1/g/sims/beckermr/tinkers_emu/TestBox00%d-00%d/',
                                       '/nfs/slac/g/ki/ki22/cosmo/beckermr/tinkers_emu/TestBox00%d-00%d/',
                                       '/nfs/slac/g/ki/ki23/des/beckermr/tinkers_emu/TestBox00%d-00%d/'],
                     'sherlock': ['/home/users/swmclau2/scratch/NewTrainingBoxes/TestBox0%02d-00%d/',
                                  '/home/users/swmclau2/scratch/NewTrainingBoxes/TestBox0%02d-00%d/',
                                  '/home/users/swmclau2/scratch/NewTrainingBoxes/TestBox0%02d-00%d/']} #all the same for Sherlock
        locations['long'] = locations['ki-ls']
        assert system in locations
        #loc = locations[system][0]
        loc_list = locations[system]

        if boxno < 2:
            loc = loc_list[0]%(boxno, realization)
        elif 2 <= boxno < 4:
            loc = loc_list[1]%(boxno, realization)
        else:
            loc = loc_list[2]%(boxno, realization)

        gadget_loc = loc + 'output/'
        loc += 'halos/m200b/'

        tmp_fnames = ['outbgc2_%d.list' % i for i in xrange(10)]
        #tmp_fnames = ['TestBox00%d-000_out_parents_5.list' % boxno]
        tmp_scale_factors = [0.25, 0.333, 0.5, 0.540541, 0.588235, 0.645161, 0.714286, 0.8, 0.909091, 1.0]
        #tmp_scale_factors = [0.645161]
        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        version_name = 'most_recent_%02d_%d'%(boxno, realization)

        super(TestBox, self).__init__(simname=simname, loc=loc, columns_to_keep=columns_to_keep, Lbox=Lbox,
                                      pmass=pmass, cosmo=cosmo, version_name = version_name, gadget_loc=gadget_loc,**new_kwargs)

        cache_locs = {'ki-ls': '/u/ki/swmclau2/des/halocats/hlist_%.2f.list.%s_%02d_%d.hdf5',
                      'sherlock': '/scratch/users/swmclau2/halocats/hlist_%.2f.list.%s_%02d_%d.hdf5'}
        cache_locs['long'] = cache_locs['ki-ls']
        self.cache_loc = path.dirname(cache_locs[system])#%(a, self.simname, boxno)
        self.cache_filenames = [cache_locs[system] % (a, self.simname, boxno, realization )
                                for a in self.scale_factors]  # make sure we don't have redunancies.

    def _get_cosmo(self):
        """
        Construct the cosmology object taht corresponds to this boxnumber
        :param boxno:
            The box number whose cosmology we want
        :return:
            Cosmology, the FlatwCDM astropy cosmology with the parameters of boxno
        """
        params = self.cosmo_params.iloc[self.boxno]
        h = params['H0'] / 100.0
        Om0 = (params['ombh2'] + params['omch2']) / (h ** 2)
        return cosmology.core.FlatwCDM(H0=params['H0'], Om0=Om0, Neff=params['Neff'],
                                       Ob0=params['ombh2'] / (h ** 2))

    def _get_cosmo_param_names_vals(self):
        # TODO docs
        names = list(self.cosmo_params.columns.values)
        vals = np.array(list(self.cosmo_params.iloc[self.boxno].values))
        return names, vals

class FastPM(Cat):

    def __init__(self, boxno, system='ki-ls', **kwargs):

        assert 0<=boxno<=121
        assert int(boxno) == boxno

        self.boxno = boxno

        simname = 'fastpm'
        colums_to_keep = {'halo_id': (0,'i8'), 'halo_mvir': (1,'f4'),
                           'halo_x':(2,'f4'), 'halo_y':(3,'f4'), 'halo_z':(4,'f4')}
        #Lbox = 1000.0
        self.npart = 1024 #can generalize these from the file

        if system=='ki-ls':
            fname = "/nfs/slac/g/ki/ki18/des/swmclau2/DES_emu.hdf5"
            #raise NotImplementedError("File not on ki-ls")
        else: #sherlock
            fname = "/scratch/users/swmclau2/DES_emu.hdf5"

        self.fname = fname

        f = h5py.File(fname, 'r')
        pmasses = np.array(f.attrs['pmasses'])
        pmass = pmasses[boxno]
        cfg = literal_eval(f.attrs['cfg_info'])
        Lbox = cfg['boxsize']

        self.cosmo_params = pd.read_csv(cfg['lhc_fname'], sep=' ', index_col=None)
        f.close()
        cosmo = self._get_cosmo()

        tmp_scale_factors = [ 0.645, 1.0]
        tmp_fnames = ['tmpA','tmpB']

        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)
        self.prim_haloprop_key='halo_mvir'

        version_name = 'most_recent_%02d'%boxno

        if 'simname' in kwargs:
            del kwargs['simname']

        super(FastPM, self).__init__(simname=simname,Lbox=Lbox,pmass=pmass,halo_finder='FOF',
                                     version_name=version_name, cosmo=cosmo, **kwargs)

        cache_locs = {'ki-ls': '/u/ki/swmclau2/des/halocats/hlist_%.2f.list.%s_%03d.hdf5',
                      'sherlock': '/scratch/users/swmclau2/halocats/hlist_%.2f.list.%s_%03d.hdf5'}
        cache_locs['long'] = path.dirname(cache_locs['ki-ls'])
        self.cache_loc = path.dirname(cache_locs[system])  # %(a, self.simname, boxno)
        self.cache_filenames = [cache_locs[system] % (a, self.simname, boxno)
                                for a in self.scale_factors]  # make sure we don't have redunancies.


    def _get_cosmo(self):

        params = self.cosmo_params.iloc[self.boxno]
        return cosmology.core.FlatwCDM(H0=params['h0']*100, Om0 = params['Omega_m'],\
                                         Neff=params['Neff'], Ob0=params['Omega_b'], w0 = params['w'])

    def _get_cosmo_param_names_vals(self):
        names = list(self.cosmo_params)#.values)
        vals = np.array(list(self.cosmo_params.iloc[self.boxno].values))
        return names, vals

    def _rvir_from_mvir(self, halo_mvir, z):
        rho_c = self.cosmology.critical_density(z)
        halo_mvir = halo_mvir*u.solMass
        return np.cbrt(halo_mvir*3/(4*np.pi*200*rho_c)).to('Mpc').value/self.h

    def cache(self, scale_factors = 'all', overwrite=False, **kwargs):

        for bad_kwarg in ['add_local_density', 'add_particles', 'downsample_factor']:
            if bad_kwarg in kwargs:
                raise NotImplementedError("Particles not allowed with FastPM")

        f = h5py.File(self.fname, 'r')
        grp = f['Box_%03d'%self.boxno]
        for a, cache_fnames in izip(self.scale_factors, self.cache_filenames):
            if scale_factors != 'all' and a not in scale_factors:
                continue
            z = 1./a -1.0
            z_key = 'z=%0.3f'%z
            data = grp[z_key]['halos'].value
            # assuming halo_columns
            # TODO do this better
            # problem with the ids, not uinique? don't understand.
            halo_id = np.array(range(len(data)))#data[:, 0]
            halo_upid = np.ones_like(halo_id)*-1
            halo_mass = data[:, 1]
            halo_rvir = self._rvir_from_mvir(halo_mass, z)
            halo_nfw_conc = np.ones_like(halo_mass)
            halo_x, halo_y, halo_z = data[:, 2], data[:, 3], data[:, 4]
            halo_vx= halo_vy= halo_vz = np.zeros_like(halo_x)
            for halo_pos in [halo_x, halo_y, halo_z]:
                halo_pos[halo_pos<0] = self.Lbox - halo_pos[halo_pos<0]
                halo_pos[halo_pos>self.Lbox] = halo_pos[halo_pos>self.Lbox] - self.Lbox

            halocat = UserSuppliedHaloCatalog(redshift=z, Lbox=self.Lbox, particle_mass=self.pmass,
                                              halo_id=halo_id,halo_upid=halo_upid, halo_mvir=halo_mass,halo_nfw_conc=halo_nfw_conc,
                                              halo_rvir=halo_rvir,
                                              halo_x=halo_x, halo_y=halo_y, halo_z=halo_z,
                                              halo_vx=halo_vx, halo_vy=halo_vy, halo_vz=halo_vz)

            halocat.add_halocat_to_cache(fname=cache_fnames, processing_notes = "FastPM with FOF halos",
                                         simname=self.simname, halo_finder='FOF',
                                         version_name=self.version_name, overwrite=overwrite)

    def _read_particles(self, snapdir, downsample_factor):
        raise NotImplementedError

    def cache_particles(self, particles, scale_factor, downsample_factor):
        raise NotImplementedError

class DarkSky(Cat):
    def __init__(self, boxno, system = 'sherlock', **kwargs):
        allowed_boxnos = set([''.join(j) for j in product(''.join([str(i) for i in xrange(8)]), repeat = 3)])
        assert int(boxno) == boxno
        assert '%03d'%boxno in allowed_boxnos
        self.boxno = int(boxno)

        simname = 'ds14'
        columns_to_keep = {'halo_id': (6, 'i8'),'halo_upid':(7, 'i8'), 'halo_mvir': (3, 'f4'),
                          'halo_x': (0, 'f4'), 'halo_y': (1, 'f4'), 'halo_z': (2, 'f4'),
                          'halo_vmax': (5, 'f4'), 'halo_rvir':(4, 'f4')}
        Lbox = 1000.0
        #self.npart = 1024  # can generalize these from the file

        if system == 'sherlock':
            # TODO update
            fname = '/oak/stanford/orgs/kipac/users/swmclau2/Darksky/ds14_a_halos_1.0000.hdf5'
            ptcl_fname = '/scratch/users/swmclau2/Darksky/ds14_a_1.0000_0.010_downsample_v3.hdf5'
        else:  # ki-ls, etc
            raise NotImplementedError("File not on ki-ls")
        self.fname = fname
        self.ptcl_fname = ptcl_fname

        pmass = 5.6749434e10 #Msun  (/h?)

        self.cosmo_params = {'Omega0_m' : 0.295037918703847,
                             'Omega0_cdm' : 0.2482735987535057,
                            'Omega0_b' : 0.04676431995034128,
                            'w0' : -1,
                            'h' : 0.7036893781978598}


        cosmo = self._get_cosmo()

        tmp_scale_factors = [1.0]
        tmp_fnames = ['tmpA']

        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)
        self.prim_haloprop_key = 'halo_mvir'

        version_name = 'most_recent_%02d' % boxno

        if 'simname' in kwargs:
            del kwargs['simname']

        super(DarkSky, self).__init__(simname=simname, Lbox=Lbox, pmass=pmass,columns_to_keep=columns_to_keep, halo_finder='rockstar',
                                     version_name=version_name, cosmo=cosmo, **kwargs)

        cache_locs = {'sherlock': '/scratch/users/swmclau2/halocats/hlist_%.2f.list.%s_%03d.hdf5'}
        self.cache_loc = path.dirname(cache_locs[system])  # %(a, self.simname, boxno)
        self.cache_filenames = [cache_locs[system] % (a, self.simname, boxno)
                                for a in self.scale_factors]  # make sure we don't have redunancies.

    def _get_cosmo(self):
        return cosmology.core.FlatLambdaCDM(H0=self.cosmo_params['h'] * 100, Om0=self.cosmo_params['Omega0_m'], \
                                       Ob0=self.cosmo_params['Omega0_b'])#, w0=params['w'])

    def _get_cosmo_param_names_vals(self):
        names = ['Omega_c', 'Omega_b','w0', 'h', 'sigma8', 'n_s']

        vals = [0.2482735987535057, 0.04676431995034128, -1, self.h, 0.835, 0.967]

        return names, vals


    def cache(self, scale_factors = 'all', overwrite=False, **kwargs):

        # TODO particles
        for bad_kwarg in ['add_local_density', 'add_particles', 'downsample_factor']:
            if bad_kwarg in kwargs:
                raise NotImplementedError("Particles not allowed with Darksky")

        f = h5py.File(self.fname, 'r')
        dset = f['halos']['subbox_%03d'%self.boxno]
        for a, cache_fnames in izip(self.scale_factors, self.cache_filenames):
            if scale_factors != 'all' and a not in scale_factors:
                continue
            z = 1./a -1.0
            data = dset.value
            # assuming halo_columns
            # TODO do this better
            # problem with the ids, not uinique? don't understand.
            halo_id = data['id']
            halo_upid = data['pid']
            halo_mass = data['m200b']
            halo_rvir = data['r200b']
            halo_vmax = data['vmax']
            halo_x, halo_y, halo_z = data['x']%self.Lbox, data['y']%self.Lbox , data['z']%self.Lbox

            halocat = UserSuppliedHaloCatalog(redshift=z, Lbox=self.Lbox, particle_mass=self.pmass,
                                              halo_id=halo_id,halo_upid=halo_upid, halo_mvir=halo_mass,
                                              halo_rvir=halo_rvir, halo_vmax = halo_vmax,
                                              halo_x=halo_x, halo_y=halo_y, halo_z=halo_z)
            add_halo_hostid(halocat.halo_table)

            halocat.add_halocat_to_cache(fname=cache_fnames, processing_notes = "DS14a subbox",
                                         simname=self.simname, halo_finder='rockstar',
                                         version_name=self.version_name, overwrite=overwrite)
        f.close()

    def _conc_from_mass(self, halo_mass):
        raise NotImplementedError
        # Tis is too slow to be useful.
        #params = {'flat':True, 'H0': self.cosmo_params['h'] * 100, 'Om0': self.cosmo_params['Omega0_m'],\
        #          'Ob0':self.cosmo_params['Omega0_b'],'sigma8': 0.8355, 'ns': 0.9688}
        #colossus_cosmo.setCosmology('DarkSky', params)
        #return concentration.concentration(halo_mass, 'vir', 0.0)

    def load_catalog_no_cache(self, scale_factor, min_ptcl = 50, tol=0.05, check_sf=True, particles=False):
            '''
            Load a catalog without caching. I *think* thins will work...
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
            f = h5py.File(self.fname, 'r')
            sys.stdout.flush()
            dset = f['halos']['subbox_%03d'%self.boxno]
            # only one sf so skipping some of this
            # this copies a lot from cache, could make this one function...
            data = dset[()]
            f.close() 
            halo_id = data['id']
            halo_upid = data['pid']
            halo_mass = data['m200b']
            halo_rvir = data['r200b']
            halo_vmax = data['vmax']
            halo_x, halo_y, halo_z = data['x']%self.Lbox, data['y']%self.Lbox , data['z']%self.Lbox
            halo_vx, halo_vy, halo_vz = np.zeros_like(halo_x), np.zeros_like(halo_x), np.zeros_like(halo_x)
            self.halocat = UserSuppliedHaloCatalog(redshift=z, Lbox=self.Lbox, particle_mass=self.pmass,
                                              halo_id=halo_id,halo_upid=halo_upid, halo_mvir=halo_mass,
                                              #halo_nfw_conc = halo_nfw_conc,
                                              halo_rvir=halo_rvir, halo_vmax = halo_vmax,
                                              halo_x=halo_x, halo_y=halo_y, halo_z=halo_z,
                                              halo_vx=halo_vx, halo_vy=halo_vy, halo_vz=halo_vz)

            add_halo_hostid(self.halocat.halo_table)
            # refelct the current catalog
            self.z = z
            self.a = a
            self.populated_once = False  # no way this one's been populated!

            if particles: # now do the particles
                assert hasattr(self, "ptcl_fname"), "Darksky has no particle file."

                f = h5py.File(self.ptcl_fname, 'r')
                dset = f['particles']['subbox_%03d'%self.boxno]
                # only one sf so skipping some of this
                # this copies a lot from cache, could make this one function...
                sys.stdout.flush()
                p_x, p_y, p_z = np.zeros((len(dset),)), np.zeros((len(dset),)), np.zeros((len(dset),)) 

                last = 0
                step_size = int(1e6)
                for step in np.arange(step_size, len(dset), step_size):
                    p_x[last:last+step] = dset[last:last+step, 0]%self.Lbox
                    p_y[last:last+step] = dset[last:last+step, 1]%self.Lbox
                    p_z[last:last+step] = dset[last:last+step, 2]%self.Lbox

                    last = step

                #p_x = dset[:, 0]%self.Lbox
                #p_y = dset[:, 1]%self.Lbox
                #p_z = dset[:, 2]%self.Lbox
                f.close() 
                # fix bounds cuz some are still off?
                def fix_bounds(x):
                    x_lt_0 = x< 0
                    x_gt_L = x>self.Lbox
                    x[x_lt_0] =  x[x_lt_0]+self.Lbox
                    x[x_gt_L] = x[x_gt_L] - self.Lbox
                    return x

                #p_x = fix_bounds(p_x)
                #p_y = fix_bounds(p_y)
                #p_z = fix_bounds(p_z)

                vx=vy=vz = np.zeros_like(p_x)
                print 'HH'
                ptcl_ids = np.array(range(p_x.shape[0]))

                ptcl_catalog = UserSuppliedPtclCatalog(redshift=z, Lbox=self.Lbox, particle_mass=self.pmass, x=p_x, y=p_y, z=p_z, vx=vx, vy=vy, vz=vz, ptcl_ids=ptcl_ids)
                print 'II'
                # attach this to the halocat
                setattr(self.halocat, "ptcl_table", ptcl_catalog.ptcl_table)

    def load_model(self, scale_factor, HOD='redMagic', check_sf=True, hod_kwargs={}):
        '''
        Load an HOD model. Not reccomended to be used separately from the load function. It
        is possible for the scale_factor of the model and catalog to be different.

        Copied from the cat form, to implement mass_conc relation
        would be nice to do this in a cleaner way. if i have to more than once..
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
            assert HOD in VALID_HODS
            if HOD in VALID_HODS - DEFAULT_HODS:  # my custom ones
                cens_occ = HOD_DICT[HOD][0](redshift=z, **hod_kwargs)
                # TODO  this is a hack, something better would be better
                try:  # hack for central modulation
                    # the ab ones need to modulated with the baseline model
                    sats_occ = HOD_DICT[HOD][1](redshift=z, cenocc_model=cens_occ, **hod_kwargs)
                except:  # assume the error is a cenocc issue
                    sats_occ = HOD_DICT[HOD][1](redshift=z, **hod_kwargs)

                self.model = HodModelFactory(
                    centrals_occupation=cens_occ,
                    centrals_profile=TrivialPhaseSpace(redshift=z),
                    satellites_occupation=sats_occ,
                    satellites_profile=NFWPhaseSpace(redshift=z, conc_mass_model = 'dutton_maccio14'))

            else:
                raise ValueError("This won't work with darksky, sorry!")
                self.model = PrebuiltHodModelFactory(HOD, redshift=z, **hod_kwargs)
                #self.model.model_dictionary['satellites_profile'].conc_mass_model = 'dutton_maccio14'
        else:
            cens_occ = HOD[0](redshift=z, **hod_kwargs)
            # NOTE don't know if should always modulate, but I always do.
            sats_occ = HOD[1](redshift=z, cenocc_model=cens_occ, **hod_kwargs)
            self.model = HodModelFactory(
                centrals_occupation=cens_occ,
                centrals_profile=TrivialPhaseSpace(redshift=z),
                satellites_occupation=sats_occ,
                satellites_profile=NFWPhaseSpace(redshift=z, conc_mass_model = 'dutton_maccio14'))

        self.populated_once = False  # cover for loadign new ones

class ResolutionTestBox(Cat):


    def __init__(self, boxno, system='ki-ls', **kwargs):

        #assert realization == 0
        #assert 0 <= boxno <= 2
        assert boxno in set([0, 4,5,6,7,8,9,10])
        cosmo = cosmology.core.LambdaCDM(H0=100 * 0.7, Om0=0.286, Ode0=0.714)

        simname = 'resolution'# not sure if something with Emu would work better, but wanna separate from Emu..
        columns_to_keep = OUTLIST_BGC2_COLS.copy()
        del columns_to_keep['halo_m200b']
        columns_to_keep['halo_mvir'] = (2, 'f4')
        del columns_to_keep['halo_r200b']
        columns_to_keep['halo_rvir'] = (5, 'f4')

        npart_dict = {0: 535,
                      4: 1071,
                      5: 535,
                      6: 1071,
                      7: 535,
                      8: 1071,
                      9: 535,
                      10: 1071}
        Lbox =  400.0 # Mpc/h
        self.npart = npart_dict[boxno]
        # Need to make a way to combine all the params

        pmass = 3.29908e10 * (535.0/self.npart)**3

        self.prim_haloprop_key = 'halo_m200b'
        locations = {'ki-ls': '/u/ki/jderose/ki18/jderose/tinkers_emu/Box0%02d/halos/m200b/'}
        assert system in locations
        loc = locations[system]%boxno

        tmp_fnames = ['outbgc2_%d.list' % i for i in xrange(10)]
        #tmp_fnames = ['TestBox00%d-000_out_parents_5.list' % boxno]
        tmp_scale_factors = [0.25, 0.333, 0.5, 0.540541, 0.588235, 0.645161, 0.714286, 0.8, 0.909091, 1.0]
        #tmp_scale_factors = [0.645161]
        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]
        
        version_name = 'most_recent_%02d'%boxno

        super(ResolutionTestBox, self).__init__(simname=simname, loc=loc, columns_to_keep=columns_to_keep, Lbox=Lbox,
                                      pmass=pmass, cosmo=cosmo,version_name = version_name, **new_kwargs)

        cache_locs = {'ki-ls': '/u/ki/swmclau2/des/halocats/hlist_%.2f.list.%s_%02d.hdf5',
                      'sherlock': '/scratch/users/swmclau2/halocats/hlist_%.2f.list.%s_%02d.hdf5'}
        cache_locs['long'] = cache_locs['ki-ls']
        self.cache_filenames = [cache_locs[system] % (a, self.simname, boxno)
                                for a in self.scale_factors]  # make sure we don't have redunancies.

    def _get_cosmo_param_names_vals(self):
        # TODO docs
        names = ['Omega_m', 'Omega_b','h', 'sigma8', 'n_s']
        vals = [0.286, 0.047,self.h, 0.82, 0.96]
        return names, vals

class MDPL2(Cat):

    def __init__(self, system='sherlock', **kwargs):

        simname = 'mdpl2'
        columns_to_keep = HLIST_COLS

        # halotools only likes mvir columns
        # load m200b instead of mvir to be consistent with the train/test boxes
        del columns_to_keep['halo_m200b']
        columns_to_keep['halo_mvir'] = (39, 'f4')
        del columns_to_keep['halo_rvir'] # TODO gonna have to compute this directly
        #columns_to_keep['halo_rvir'] = (5, 'f4')

        Lbox = 1000.0 # Mpc/h
        self.npart = 3840


        pmass = 1.51e9 # Msun/h
        cosmo = self._get_cosmo()

        self.prim_haloprop_key = 'halo_m200b'

        locations = {'cori': '/global/cscratch1/sd/swmclau2/MDPL2/',
                     'sherlock':'/scratch/users/swmclau2/MDPL2/'}

        assert system in locations, "MDPL2 currently only on Cori and Sherlock"
        loc = locations[system]

        gadget_loc = loc #path.join(loc, 'snapdir_130/')

        tmp_fnames = ['hlist_1.00000.list']
        tmp_scale_factors = [1.0]

        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs = kwargs.copy()

        for key in ['simname', 'loc', 'columns_to_keep', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        cache_loc = '/scratch/users/swmclau2/hlists/'
        super(MDPL2, self).__init__(simname=simname, loc=loc, columns_to_keep=columns_to_keep, Lbox=Lbox,
                                          pmass=pmass, cosmo=cosmo, gadget_loc=gadget_loc, cache_loc=cache_loc,
                                          **new_kwargs)

    def _get_cosmo(self):
        h = 0.6777
        Om0 = 0.307115
        Ob0 = 0.048206
        #n_s = 0.96
        #sigma8 = 0.8228

        return cosmology.core.FlatLambdaCDM(H0 = h*100, Om0=Om0, Ob0=Ob0)

    def cache(self, scale_factors='all', overwrite=False, add_local_density=False, add_particles=False,
              downsample_factor=1e-2):
        '''
        A copy of the main cat version of this object. does the same thing, but
        also adds the missing r200b column before cacheing as rvir.
        :param scale_factors:
        :param overwrite:
        :param add_local_density:
        :param add_particles:
        :param downsample_factor:
        :return:
        '''
        try:
            assert not add_local_density or self.gadget_loc  # gadget loc must be nonzero here!
            assert not add_particles or self.gadget_loc
        except:
            raise AssertionError(
                'Particle location not specified; please specify gadget location for %s' % self.simname)

        if add_local_density or add_particles:
            all_snapdirs = sorted(glob(path.join(self.gadget_loc, 'snapdir*')))
            snapdirs = [all_snapdirs[idx] for idx in
                        self.sf_idxs]  # only the snapdirs for the scale factors were interested in .
        else:
            snapdirs = ['' for i in self.scale_factors]

        for a, z, fname, cache_fnames, snapdir in izip(self.scale_factors, self.redshifts, self.filenames,
                                                       self.cache_filenames, snapdirs):
            # TODO get right reader for each halofinder.
            print a, z
            if scale_factors != 'all' and a not in scale_factors:
                continue
            reader = RockstarHlistReader(fname, self.columns_to_keep, cache_fnames, self.simname,
                                         self.halo_finder, z, self.version_name, self.Lbox, self.pmass,
                                         overwrite=overwrite)
            reader.read_halocat(self.columns_to_convert)

            ###### New stuff ####
            # actually r200b, but halotools can have issues with other mass defs
            reader.halo_table['halo_rvir'] = np.cbrt( (const.G*const.M_sun/(100*u.km/u.s/u.Mpc)**2)*reader.halo_table['halo_mvir']/100).to('Mpc').value
            #####################
            if add_local_density or add_particles:
                particles = self._read_particles(snapdir, downsample_factor=downsample_factor)
                print 'sneaky', particles.shape
                if add_local_density:
                    self.add_local_density(reader, particles, downsample_factor)  # TODO how to add radius?

            reader.write_to_disk()  # do these after so we have a halo table to work off of
            reader.update_cache_log()

            if add_particles:
                self.cache_particles(particles, a, downsample_factor=downsample_factor)


# a dict that maps the default simnames to objects, which makes construction easier.
# TODO kitten_dict
cat_dict = {'bolshoi': Bolshoi, 'multidark': Multidark, 'emu': Emu, 'fox': Fox, 'multidark_highres': MDHR,
            'chinchilla': Chinchilla, 'aardvark': Aardvark, 'guppy': Guppy,
            'trainingbox': TrainingBox, 'testbox': TestBox, 'fastpm':FastPM, 'resolution': ResolutionTestBox,
            'mdpl2': MDPL2}
