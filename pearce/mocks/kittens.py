#!/bin/bash
'''Module with a clever name that holds all the custom subclasses of  Cat. Each is essentially
a holder for information to simplify the process of making Cats. It takes care of a lot of file handling/
busy work.

Each takes two kwargs: "filenames" and "scale_factors", both lists which allow the user to restrict
which particular files are cached/loaded/etc. '''

from glob import glob
from astropy import cosmology
from .cat import Cat

__all__ = ['Bolshoi', 'Multidark', 'Emu', 'Fox', 'MDHR', 'Chinchilla', 'Aardvark', 'Guppy', 'cat_dict']

#Different table schema.
HLIST_COLS = {'halo_id': (1, 'i8'), 'halo_upid': (6, 'i8'),
              'halo_x': (17, 'f4'), 'halo_y': (18, 'f4'), 'halo_z': (19, 'f4'),
              'halo_vx': (20, 'f4'), 'halo_vy': (21, 'f4'), 'halo_vz': (22, 'f4'),
              'halo_mvir': (10, 'f4'), 'halo_rvir': (11, 'f4'), 'halo_rs': (12, 'f4')}

OUTLIST_COLS = {'halo_id': (0, 'i8'), 'halo_upid': (36, 'i8'),
                'halo_x': (8, 'f4'), 'halo_y': (9, 'f4'), 'halo_z': (10, 'f4'),
                'halo_vx': (11, 'f4'), 'halo_vy': (12, 'f4'), 'halo_vz': (13, 'f4'),
                'halo_mvir': (2, 'f4'), 'halo_rvir': (5, 'f4'), 'halo_rs': (6, 'f4')}

#Previously, I had Lbox and Npart be required for all, even tho it isn't actually. I don't know why now...
#I've also removed the updating of the kwargs with the defaults.
#The user shouldn't be updating thee cosmology, simname, etc. The only thing they can change
#are the scale factors that are loaded/cached (or their filenames). The only other thing
#maybe worth changing is hlist_cols, but I find that unlikely.
class Aardvark(Cat):
    # Lbox technically required, but I don't even have access to anything besides 400. Ignore for now.
    def __init__(self, system='ki-ls', **kwargs):

        #set these defaults
        simname='aardvark'
        cosmo = cosmology.core.LambdaCDM(H0=100 * 0.73, Om0=0.23, Ode0=0.77, Ob0=0.047)
        pmass = 4.75619e+08
        Lbox = 400.0
        columns_to_keep = HLIST_COLS

        locations = {'ki-ls': '/nfs/slac/g/ki/ki18/des/mbusha/simulations/Aardvark-2PS/Lb400/rockstar/hlists/'}
        assert system in locations
        loc = locations[system]

        tmp_fnames = glob(loc+ 'hlist_*.list')  # snag all the hlists
        tmp_fnames = [fname[len(loc):] for fname in tmp_fnames]  # just want the names in the dir
        tmp_scale_factors = [float(fname[6:-5]) for fname in tmp_fnames]  # pull out scale factors

        # If use specified scale factors or filenames, select only those.
        # Looked into a way to put this in the global init.
        # However, the amount of copy-pasting that would save would be minimal, it turns out.
        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)
        #Now fnames and scale factors are rolled into kwargs
        new_kwargs = kwargs.copy()
        #delete the keys that are fixed, if they exist
        for key in ['simname', 'loc', 'columns_to_keep', 'Lbox','pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        super(Aardvark, self).__init__(simname, loc,columns_to_keep=columns_to_keep,Lbox=Lbox,\
                                       pmass=pmass,cosmo=cosmo, **new_kwargs)

class Bolshoi(Cat):
    'Builtin catalog, so needs very little. Will never be manually cached!'

    def __init__(self):
        simname = 'bolshoi'
        version_name = 'halotools_alpha_version2'
        scale_factors = [1.0]

        super(Bolshoi,self).__init__(simname, version_name=version_name, scale_factors=scale_factors)

class Chinchilla(Cat):
    'The most complicated (and biggest) subclass'
    def __init__(self, Lbox, npart=2048, system='ki-ls', **kwargs):
        #Lbox required, npart should be if Lbox is not 1050!
        simname = 'chinchilla'
        cosmo = cosmology.core.LambdaCDM(H0=100 * 0.7, Om0=0.286, Ode0=0.714)

        if Lbox not in {125.0, 250.0, 400.0, 1050.0}:
            raise ValueError("Invalid boxsize %.1f for Chinchilla"%Lbox)


        #the different box sizes are in differenct locations, among other things.
        if Lbox==1050:
            locations = {'ki-ls':'/u/ki/swmclau2/des/rockstar_outputs/'}
            assert system in locations #necessary?
            loc = locations[system]

            columns_to_keep = OUTLIST_COLS
            pmass = 3.34881e+10
            npart = 1400
            version_name = 'Lb1050-1400'

            tmp_scale_factors = [0.1429, 0.1667, 0.2, 0.25, 0.3333, 0.4, 0.5, 0.6667, 0.8, 1.0]
            tmp_fnames = ['out_%d.list' % i for i in xrange(10)]
        else:
            locations = {'ki-ls': '/nfs/slac/g/ki/ki21/cosmo/yymao/sham_test/resolution-test/',
                         'sherlock':'/scratch/users/swmclau2/hlists/Chinchilla/'}

            assert system in locations
            loc = locations[system]
            columns_to_keep = HLIST_COLS
            #may also work for 1050 too.
            pmass = 1.44390e+08*((Lbox / 125.0) ** 3) * ((1024.0/npart)** 3)

            valid_version_names = {'Lb125-1024', 'Lb125-2048', 'Lb250-1024', 'Lb250-128',
                                    'Lb250-196', 'Lb250-2048', 'Lb250-2560', 'Lb250-320',
                                    'Lb250-512', 'Lb250-768', 'Lb250-85', 'Lb400-1024',
                                    'Lb400-136', 'Lb400-2048', 'Lb400-210', 'Lb400-315',
                                    'Lb400-512', 'Lb400-768'}

            #check that this combination of lbox and npart exists
            version_name = 'Lb%d-%d' % (int(Lbox), npart)
            assert version_name in valid_version_names
            #add a subdirectory
            loc += 'c%d-%d/' % (int(Lbox), int(npart))

            if system=='ki-ls':  # differences in how the files are stored.
                loc += '/rockstar/hlists/'


            tmp_fnames = glob(loc + 'hlist_*.list')  # snag all the hlists
            tmp_fnames = [fname[len(loc):] for fname in tmp_fnames]  # just want the names in the dir
            tmp_scale_factors = [float(fname[6:-5]) for fname in tmp_fnames]  # pull out scale factors

        # if the user passed in stuff, have to check a bunch of things
        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs =kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep','version_name', 'Lbox','pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        super(Chinchilla, self).__init__(simname=simname, loc=loc, columns_to_keep = columns_to_keep,
                                         version_name=version_name, Lbox=Lbox,
                                         pmass=pmass, cosmo=cosmo, **new_kwargs)
        #Chinchillas also have to be cached differently.
        cache_locs = {'ki-ls':'/u/ki/swmclau2/des/halocats/hlist_%.2f.list.%s_%s.hdf5',
                      'sherlock':'/scratch/users/swmclau2/halocats/hlist_%.2f.list.%s_%s.hdf5' }
        self.cache_locs = [cache_locs['system'] % (a, self.simname, self.version_name)
                           for a in self.scale_factors]  # make sure we don't have redunancies.

class Emu(Cat):

    def __init__(self, Lbox, system='ki-ls', **kwargs):

        simname = 'emu'

        # The emu naming convention is different than the others; two Emu boxes of the same number don't have the same
        #cosmology. This will need to be expanded later.
        if Lbox == 1050.0:
            locations = {'ki-ls':'/u/ki/swmclau2/des/emu/Box000/',
                         'sherlock': '/scratch/users/swmclau2/hlists/emu/Box000/' }
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

            tmp_fnames = glob(loc+ 'hlist_*.list')  # snag all the hlists
            tmp_fnames = [fname[len(kwargs['loc']):] for fname in tmp_fnames]  # just want the names in the dir
            tmp_scale_factors = [float(fname[6:-5]) for fname in tmp_fnames]

        version_name = 'most_recent_%d'%int(Lbox)

        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)
        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'version_name', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        super(Emu, self).__init__(simname=simname, loc=loc, columns_to_keep=columns_to_keep,
                                  version_name=version_name,Lbox=Lbox,pmass=pmass,
                                  cosmo=cosmo,**new_kwargs)

class Fox(Cat):
    def __init__(self, system='ki-ls', **kwargs):
        simname = 'fox'
        columns_to_keep = HLIST_COLS
        Lbox = 400.0 #only one available
        pmass =  6.58298e8
        cosmo = cosmology.core.LambdaCDM(H0=100 * 0.6704346, Om0=0.318340, Ode0=0.681660)
        locations = {'ki-ls':'/nfs/slac/g/ki/ki23/des/BCCSims/Fox/Lb400/halos/rockstar/output/hlists/',
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

        super(Fox,self).__init__(simname=simname, loc=loc,columns_to_keep=columns_to_keep, Lbox=Lbox,
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
                                  pmass=pmass,cosmo=cosmo, **kwargs)

class Multidark(Cat):
    'Builtin, so needs very little. Will never be cached!'

    def __init__(self, **kwargs):
        simname = 'multidark'
        version_name = 'halotools_alpha_version2'
        scale_factors = [2.0/3, 1.0]

        super(Multidark, self).__init__(simname, version_name=version_name, scale_factors=scale_factors)

#TODO put this under Multidark? not sure I want to mix builtins and mine.
class MDHR(Cat):
    def __init__(self, system='ki-ls', **kwargs):

        simname = 'multidark_highres'
        columns_to_keep = HLIST_COLS
        Lbox = 1e3
        pmass = 8.721e9

        #MDHR uses the default WMAP cosmology

        locations = {'ki-ls':'/nfs/slac/g/ki/ki20/cosmo/behroozi/MultiDark/hlists/',
                     'sherlock':'/scratch/users/swmclau2/hlists/MDHR/' }
        assert system in locations
        loc = locations[system]

        tmp_scale_factors = [0.25690, 0.34800, 0.49990, 0.53030, 0.65180, 0.71250, 0.80370, 0.91000, 1.00110]
        tmp_fnames = ['hlist_%.5f.list' % a for a in tmp_scale_factors]

        self._update_lists(kwargs, tmp_fnames, tmp_scale_factors)

        new_kwargs = kwargs.copy()
        for key in ['simname', 'loc', 'columns_to_keep', 'version_name', 'Lbox', 'pmass', 'cosmo']:
            if key in new_kwargs:
                del new_kwargs[key]

        super(MDHR, self).__init__(simname, loc=loc,columns_to_keep=columns_to_keep, Lbox=Lbox,
                                   pmass=pmass **new_kwargs)

#a dict that maps the default simnames to objects, which makes construction easier.
#TODO kitten_dict
cat_dict = {'bolshoi': Bolshoi, 'multidark': Multidark, 'emu': Emu, 'fox': Fox, 'multidark_highres': MDHR,
            'chinchilla': Chinchilla,
            'aardvark': Aardvark, 'guppy': Guppy}
