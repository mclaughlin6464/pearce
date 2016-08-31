#!/bin/bash
'''Module with a clever name that holds all the custom subclasses of  Cat.'''

from astropy import cosmology
from cat import Cat

__all__ = ['Bolshoi', 'Multidark', 'Emu', 'Fox', 'MDHR', 'Chinchilla', 'Aardvark', 'Guppy', 'cat_dict']

#Previously, I had Lbox and Npart be required for all, even tho it isn't. I don't know why now...
class Aardvark(Cat):
    # Lbox technically required, but I don't even have access to anything besides 400. Ignore for now.
    def __init__(self, Lbox, **kwargs):

        defaults = {'simname': 'aardvark', 'loc': DEFAULT_LOCS['aardvark'],
                    'columns_to_keep': HLIST_COLS,
                    'cosmo': cosmology.core.LambdaCDM(H0=100 * 0.73, Om0=0.23, Ode0=0.77, Ob0=0.047),
                    'pmass': 4.75619e+08, 'Lbox': 400.0}

        simname='aardvark'
        cosmo = cosmology.core.LambdaCDM(H0=100 * 0.73, Om0=0.23, Ode0=0.77, Ob0=0.047)


        # Only one boxsize available for Aardvark presently
        if Lbox == 400.0:
            pass
        else:
            raise IOError('%.1f is not a valid boxsize for Aardvark' % (Lbox))

        for key, value in defaults.iteritems():
            if key not in kwargs or kwargs[key] is None:
                kwargs[key] = value

        tmp_fnames = glob(kwargs['loc'] + 'hlist_*.list')  # snag all the hlists
        tmp_fnames = [fname[len(kwargs['loc']):] for fname in tmp_fnames]  # just want the names in the dir
        tmp_scale_factors = [float(fname[6:-5]) for fname in tmp_fnames]  # pull out scale factors

        # If use specified scale factors or filenames, select only those.
        # Looked into a way to put this in the global init.
        # However, the amount of copy-pasting that would save would be minimal, it turns out.
        self.update_lists(kwargs, tmp_fnames, tmp_scale_factors)
        super(Aardvark, self).__init__(**kwargs)