from pearce.mocks.kittens import cat_dict
import numpy as np
#TODO all redshifts
scale_factors = [0.54780, 0.59260, 0.658, 0.71170, 0.8112]

for a in scale_factors:

    z = 1.0/a - 1.0
    #load a paritcular mock, given this name, size and redshift
    cosmo_params = {'simname':'chinchilla', 'Lbox':400.0, 'scale_factors':[a]}
    cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!
    cat.load(a, tol = 0.01, HOD='hsabRedMagic', particles = True)#, hod_kwargs = {'sec_haloprop_key':'halo_log_nfw_conc'})#, hod_kwargs={'split': 0.5})

    # chain I ran awhile ago

    n_walkers = 500 #200
    # i'm running new chains at all 5 bins, this is the format they'll have when they're done 
    #chain_fname = '/u/ki/swmclau2/des/PearceMCMC/%d_walkers_%d_steps_chain_redmagic_bias_z%.2f.npy'%(200, 50000, z)
    chain_fname = '/u/ki/swmclau2/des/SherlockPearceMCMC/500_walkers_5000_steps_chain_wt_alt_redmagic_z0.23_part2.npy'
    chain = np.genfromtxt(chain_fname)

    n_params = chain.shape[1]
    n_burn = 0
    chain = chain[n_walkers*n_burn:, :]

    # the chain's column names
    ordered_param_names = ['logMmin','mean_occupation_centrals_assembias_param1', 'f_c', 'logM0', 'sigma_logM',
                                         'mean_occupation_satellites_assembias_param1',     'logM1',   'alpha']

    theta_bins = np.logspace(np.log10(2.5), np.log10(250), 21)/60 #binning used in buzzard mocks
    tpoints = (theta_bins[1:]+theta_bins[:-1])/2


    # W is the prefactor for w(theta). Can be calculated from compute_wt_prefactor in the cat object of pearce
    W = 1.0#0.00275848072207
    # sigma crit inverse is the prefactor to gt. It can be calculated from calc_sigma_crit_inv
    # I believe these should take the same syntax, but double check that your outputs are sensible.
    sc_inv = 1.0#0.000123904180342

    size = 100
    params = np.zeros((size, len(ordered_param_names)))
    wts = np.zeros((size, len(tpoints)))
    dss = np.zeros((size, len(tpoints)))

    # select a random subsample from the chain
    indicies = np.random.choice(chain.shape[0], size = size, replace = False)
    print z
    for i, row in enumerate(chain[indicies]):
        print i
        params[i] = row
        hod_params = dict(zip(ordered_param_names, row))
        # populates and calculate each observable.
        cat.populate(hod_params)
        wt = cat.calc_wt(theta_bins, W, n_cores = 4)
        ds = cat.calc_gt(theta_bins, 1.0, n_cores = 4)
        wts[i] = wt
        dss[i] = ds

    # save the chain smaples chosen and the observables we used.
    np.save("params_z_%0.3f.npy"%z, params)
    np.save("wt_z_%0.3f.npy"%z, wts)
    np.save("ds_z_%0.3f.npy"%z, dss)
