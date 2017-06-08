def run_mcmc(self, y, cov, bin_centers, nwalkers=1000, nsteps=100, nburn=20, n_cores='all'):
    """
    Run an MCMC sampler, using the emulator. Uses emcee to perform sampling.
    :param y:
        A true y value to recover the parameters of theta. NOTE: The emulator emulates some indepedant variables in
        log space, others in linear. Make sure y is in the same space!
    :param cov:
        The measurement covariance matrix of y
    :param bin_centers:
        The centers of the bins y is measured in (radial or angular).
    :param nwalkers:
        Optional. Number of walkers for emcee. Default is 1000.
    :param nsteps:
        Optional. Number of steps for emcee. Default is 100.
    :param nburn:
        Optional. Number of burn-in steps for emcee. Default is 20.
    :param n_cores:
        Number of cores, either an iteger or 'all'. Default is 'all'.
    :return:
        chain, a numpy array of the sample chain.
    """

    assert n_cores == 'all' or n_cores > 0
    if type(n_cores) is not str:
        assert int(n_cores) == n_cores

    max_cores = cpu_count()
    if n_cores == 'all':
        n_cores = max_cores
    elif n_cores > max_cores:
        warnings.warn('n_cores invalid. Changing from %d to maximum %d.' % (n_cores, max_cores))
        n_cores = max_cores
        # else, we're good!

    assert y.shape[0] == cov.shape[0] and cov.shape[1] == cov.shape[0]
    assert y.shape[0] == bin_centers.shape[0]

    sampler = mc.EnsembleSampler(nwalkers, self.sampling_ndim, lnprob,
                                 threads=n_cores, args=(self, y, cov, bin_centers))

    pos0 = np.zeros((nwalkers, self.sampling_ndim))
    # The zip ensures we don't use the params that are only for the emulator
    for idx, (p, _) in enumerate(izip(self._ordered_params, xrange(self.sampling_ndim))):
        # pos0[:, idx] = np.random.uniform(p.low, p.high, size=nwalkers)
        pos0[:, idx] = np.random.normal(loc=(p.high + p.low) / 2, scale=(p.high + p.low) / 10, size=nwalkers)

    sampler.run_mcmc(pos0, nsteps)

    # Note, still an issue of param label ordering here.
    chain = sampler.chain[:, nburn:, :].reshape((-1, self.sampling_ndim))

    return chain