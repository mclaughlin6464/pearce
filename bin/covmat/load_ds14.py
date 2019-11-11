__all__ = ['load_ds14', 'load_ds14_a', 'load_ds14_b']

from itertools import product
import numpy as np
from sdfpy import SDFRead, SDFIndex

def _fields2dtype(fields):
    return np.dtype([(f, int if f.endswith('id') else float) for f in fields])

class load_ds14:
    def __init__(self, basename, midx_level=7):
        self._basename = basename
        self._sdf_halos = SDFIndex(SDFRead(self._basename), \
                SDFRead('{0}.midx{1}'.format(self._basename, midx_level)))
 
    def getfields(self):
        return self._sdf_halos.sdfdata.keys()

    def getsubboxsize(self, level=2):
        return self._sdf_halos.domain_width/float(2**level)
    
    def getsubbox(self, idx, level=2, pad=1.0, fields=None, cut=None, \
            cut_fields=None):
        """
        Return a subboxes of ds14.
        There will be 2**level subboxes on each side.
        Specify pad in Mpc/h.
        """
        try:
            pad = float(pad)
        except TypeError:
            pass
        else:
            pad = np.ones(3)*pad
            
        if fields is None:
            fields = self.getfields()
        
        if cut is not None and not hasattr(cut, '__call__'):
            raise ValueError('`cut` must be callable.')
        
        idx = tuple(idx)
        if idx not in list(product(xrange(2**level), repeat=3)):
            raise ValueError("`idx` is not correctly specified!")
        return self._loadsubbox(idx, level, pad, fields, cut, cut_fields)

    def itersubbox(self, level=2, pad=1.0, fields=None, cut=None, \
            cut_fields=None, return_index=False):
        """
        Return an iterator of the subboxes of ds14.
        There will be 2**level subboxes on each side.
        Specify pad in Mpc/h.
        """
        try:
            pad = float(pad)
        except TypeError:
            pass
        else:
            pad = np.ones(3)*pad
            
        if fields is None:
            fields = self._sdf_halos.sdfdata.keys()
        
        if cut is not None and not hasattr(cut, '__call__'):
            raise ValueError('`cut` must be callable.')
        
        for idx in product(xrange(2**level), repeat=3):
            halos = self._loadsubbox(idx, level, pad, fields, cut, cut_fields)
            yield (idx, halos) if return_index else halos
            
    def _loadsubbox(self, idx, level, pad, fields, cut, cut_fields):
        fields_dtype = _fields2dtype(fields)
        count = 0
        for h in self._sdf_halos.iter_padded_bbox_data(level, idx, pad, fields):
            if cut is None:
                flag = slice(None)
                count_this = len(h[fields[0]])
            else:
                flag = cut(*(h[f] for f in cut_fields))
                count_this = np.count_nonzero(flag)
            if count_this:
                if count:
                    data.resize(count+count_this)
                else:
                    data = np.empty(count_this, fields_dtype)
                for f in fields:
                    #print f
                    data[f][count:] = h[f][flag]
                count += count_this
        if not count:
            data = np.empty(0, fields_dtype)
        return data

class load_ds14_a(load_ds14):
    def __init__(self, midx_level=7):
        #self._basename = '/nfs/slac/g/ki/darksky/simulations/ds14_a/halos/ds14_a_halos_1.0000'
        self._basename = '/scratch/users/swmclau2/Darksky/ds14_a_halos_1.0000'

        self._sdf_halos = SDFIndex(SDFRead(self._basename), \
                SDFRead('{0}.midx{1}'.format(self._basename, midx_level)))

class load_ds14_b(load_ds14):
    def __init__(self, midx_level=7):
        self._basename = '/nfs/slac/g/ki/darksky/simulations/ds14_b/ds14_b_halos_1.0000'
        self._sdf_halos = SDFIndex(SDFRead(self._basename), \
                SDFRead('{0}.midx{1}'.format(self._basename, midx_level)))



