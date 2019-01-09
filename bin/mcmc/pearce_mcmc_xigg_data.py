# This file just makes the data for the sampler, so multiple samplers can use the exact same data.
from pearce.mocks import cat_dict
from scipy.optimize import minimize_scalar
import numpy as np
import cPickle as pickle
from os import path

fixed_params = {}#'f_c':1.0}#,'logM1': 13.8 }# 'z':0.0}

boxno, realization = 3, 0
cosmo_params = {'simname':'testbox', 'boxno': boxno, 'realization': realization, 'scale_factors':[1.0], 'system': 'sherlock'}
cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!

cat.load(1.0, HOD='zheng07')

emulation_point = [('logM0', 13.5), ('sigma_logM', 0.25), 
                    ('alpha', 0.9),('logM1', 13.5)]#, ('logMmin', 12.233)]

em_params = dict(emulation_point)
em_params.update(fixed_params)

def add_logMmin(hod_params, cat):
    """
    In the fixed number density case, find the logMmin value that will match the nd given hod_params
    :param: hod_params:
        The other parameters besides logMmin
    :param cat:
        the catalog in question
    :return:
        None. hod_params will have logMmin added to it.
    """
    hod_params['logMmin'] = 13.0 #initial guess
    #cat.populate(hod_params) #may be overkill, but will ensure params are written everywhere
    def func(logMmin, hod_params):
        hod_params.update({'logMmin':logMmin})
        return (cat.calc_analytic_nd(hod_params) - 1e-4)**2

    res = minimize_scalar(func, bounds = (12, 16), args = (hod_params,), options = {'maxiter':100}, method = 'Bounded')

    # assuming this doens't fail
    hod_params['logMmin'] = res.x

add_logMmin(em_params, cat)

r_bins = np.logspace(-1.1, 1.6, 19)
rpoints = (r_bins[1:] + r_bins[:-1])/2.0#emu.scale_bin_centers 

xi_vals = []
np.random.seed(1)
for i in xrange(50):
    cat.populate(em_params)
    xi_vals.append(cat.calc_xi(r_bins))

# TODO need a way to get a measurement cov for the shams
xi_vals = np.array(xi_vals)

cat.populate(em_params)
yjk, cov = cat.calc_xi(r_bins, do_jackknife=True, jk_args = {'n_rands':10, 'n_sub':5})
#y10 = np.loadtxt('xi_gg_true_jk.npy')
#cov10 = np.loadtxt('xi_gg_cov_true_jk.npy')

#y = np.log10(y10)
y = np.mean(xi_vals, axis = 0)

shot_cov = np.cov(xi_vals, rowvar = False)

#cov = np.log10(1+cov10/(np.outer(y10, y10))) # TODO check this is right?

np.savetxt('xi_gg_true_mean_%d%d.npy'%(boxno, realization), y)
np.savetxt('xi_gg_true_jk_%d%d.npy'%(boxno, realization), yjk)

np.savetxt('xi_gg_cov_true_jk_%d%d.npy'%(boxno, realization), cov)

#scov = np.loadtxt('xigg_scov_log.npy')

np.savetxt('xi_gg_shot_cov_true_%d%d.npy'%(boxno, realization), shot_cov)

del em_params['logMmin']
cpv = cat._get_cosmo_param_names_vals()

cosmo_param_dict = {key: val for key, val in zip(cpv[0], cpv[1])}

with open('cosmo_param_dict_%d%d.pkl'%(boxno, realization), 'w') as f:
    pickle.dump(cosmo_param_dict, f)
