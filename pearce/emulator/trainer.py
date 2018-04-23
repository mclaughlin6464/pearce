#!bin/bash
"""
New object that generates training data in a sparter way.
Rather than firing off many jobs to do the training, will do everying with MPI in one program.
Will also write to an hdf5 file, to reduce the enormous clutter produced so far.
"""
from os import path
from time import time
import warnings
from itertools import product
import yaml
import numpy as np
import pandas as pd
from mpi4py import MPI
import h5py

from pearce.mocks import cat_dict


class Trainer(object):

    def __init__(self, config_fname):
        """
        This function creates a trainer object from a YAML config
        and preapres everything necessary to run the training jobs
        :param config_fname:
            A YAML config file containing all the details needed to train
        """

        try:
            assert path.isfile(config_fname)
        except AssertionError:
            raise AssertionError("%s is not a valid filename." % config_fname)

        with open(config_fname, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

        # TODO could do more user babysitting here

        self.prep_cosmology(cfg['cosmology'])
        # TODO one thing to consider is how to have HOD-independent emu
        # quantities like xi_mm could require this.
        self.prep_hod(cfg['HOD'])
        self.prep_observation(cfg['observation'])
        self.prep_computation(cfg['computation'])

    def prep_cosmology(self, cosmo_cfg):
        """
        Prepare the cosmology catalogs we'll need.
        Identifies the relevant cat objects and stores them in a list
        Also identifies the params needed to load up specific catalogs come compute time.
        :param cosmo_cfg:
            A dictionary (loaded from the YAML file) of all parameters relevant to cosmological emulation.
            To load multiple instances of the same type of box (like, several different cosmologies in the same
            simulation set) use the argument 'boxno', which can be a list, an integer, or a colon separated pair
            like 0:40 which will be turned into a range range(0,40). If loading up one, it doesn't need to be specified.

            'simname' specifies the name of the cached cats. Other args passed in as kwargs to the __init__
            Other optional args are 'particles' and 'downsampling_factor' which determine if and which particle catalog
            to load alongside the halo catalog. Certain observables require particles, so be wary!
        """

        # if we have to load an ensemble of cats, do so. otherwise, just load up one.
        # it's also possible to load up one

        # TODO let the user specify redshifts instead? annoying.
        scale_factors = cosmo_cfg['scale_factors']
        self._scale_factors = [scale_factors] if type(scale_factors) is float else scale_factors
        cosmo_cfg['scale_factors'] = self._scale_factors #apply the same change

        if 'boxno' in cosmo_cfg:
            if ':' in cosmo_cfg['boxno']:  # shorthand to do ranges, like 0:40
                splitstr = cosmo_cfg['boxno'].split(':')
                boxnos = range(int(splitstr[0]), int(splitstr[1]))
            else:
                boxnos = cosmo_cfg['boxno']
                boxnos = [boxnos] if type(boxnos) is int else boxnos
            del cosmo_cfg['boxno']
            self.cats = []
            # if there are multiple cosmos, they need to have a boxno kwarg
            for boxno in boxnos:
                self.cats.append(
                    cat_dict[cosmo_cfg['simname']](boxno=boxno, **cosmo_cfg))  # construct the specified catalog!

        else:  # fixed cosmology
            self.cats = [cat_dict[cosmo_cfg['simname']](**cosmo_cfg)]

        self._cosmo_param_names,_ = self.cats[0]._get_cosmo_param_names_vals()
        self._cosmo_param_vals = np.zeros((len(self.cats), len(self._cosmo_param_names)))

        for idx, cat in enumerate(self.cats):
            _, vals = cat._get_cosmo_param_names_vals() 

            self._cosmo_param_vals[idx,:] = vals

        # In the future if there are more annoying params i'll turn this into a loop or something.
        # TODO does yaml parse these to bools automatically?
        self._particles = bool(cosmo_cfg['particles']) if 'particles' in cosmo_cfg else False
        self._downsample_factor = float(cosmo_cfg['downsample_factor']) if 'downsample_facor' in cosmo_cfg else 1e-3

    def prep_hod(self, hod_cfg):
        """
        Prepare HOD information we'll need.
        Stores  the HOD name and other kwargs.

        Most importantly, loads or generates the HOD parameters. Can either accept a
        'paramfile' which will be read as a pandas dataframe. The column names will be stored
        as the parameter names, and the values as the HODs to use.

        The alternative is to pass in an 'ordered_params' dictionary or ordered_dict, with
        keys as param names and values with tuples defining the lower and upper bounds of the
        parameter. Ex: ordered_params['logMmin'] = (12.5, 14.5). Also required is
        'num_hods', the number of parameters to generate. A random, non-optimized
        LHC will be generated.
        :param hod_cfg:
            A dictionary (loaded from the YAML file) of parameters relevant to HOD emulation.
            Needs to specify the name of the HOD model to be loaded; available ones are in customHODModels or halotools.
            All other args will be passed in as kwargs.
        """
        # TODO possibly better error messaging
        self.hod = hod_cfg['model']  # string defining which model
        # TODO confirm is valid?
        del hod_cfg['model']
        # A little unnerved by the assymetry between this and cosmology
        # the cosmo params are attached to the objects
        if 'paramfile' in hod_cfg:
            assert path.exists(hod_cfg['paramfile'])
            df = pd.read_csv(hod_cfg['paramfile'])
            self._hod_param_names = df.columns
            self._hod_param_vals = df.data

            del hod_cfg['paramfile']
        else:
            assert 'ordered_params' in hod_cfg
            assert 'num_hods' in hod_cfg #TODO better test here

            # make a LHC from scratch
            ordered_params = hod_cfg['ordered_params']
            self._hod_param_names = ordered_params.keys()
            self._hod_param_vals = self._make_LHC(ordered_params, hod_cfg['num_hods'])

            del hod_cfg['ordered_params']
            del hod_cfg['num_hods']

        self._hod_kwargs = hod_cfg
        # need scale factors too, but just use them from cosmology

    def _make_LHC(self, ordered_params, N):
        """Return a vector of points in parameter space that defines a latin hypercube.
            :param ordered_params:
                OrderedDict that defines the ordering, name, and ranges of parameters
                used in the trianing data. Keys are the names, value of a tuple of (lower, higher) bounds
            :param N:
                Number of points per dimension in the hypercube. Default is 500.
            :return
                A latin hyper cube sample in HOD space in a numpy array.
        """
        np.random.seed(int(time()))

        points = []
        # by linspacing each parameter and shuffling, I ensure there is only one point in each row, in each dimension.
        for plow, phigh in ordered_params.itervalues():
            point = np.linspace(plow, phigh, num=N)
            np.random.shuffle(point)  # makes the cube random.
            points.append(point)
        return np.stack(points).T

    def prep_observation(self, obs_cfg):
        """
        Prepares variables related to the calculation of the observable (xi, wp, etc).
        Must specify 'obs', the observable to be calculated. Must be a calc_<> for a cat object
        May also specify 'bins', the binning scheme for a scale dependent variable (in Mpc or degrees)
        'n_repops', the number of times to repopulate a given hod+cosmology combination
        'log_obs', whether the log of the observable should be taken
        All other arguments will be passed into the observable function as kwargs
        :param obs_cfg:
            A dictionary (from a YAML file) specifying the arguments
        """
        # required kwargs, the observable being calculated and the binning scheme along with it.
        self.obs = obs_cfg['obs']
        del obs_cfg['obs']
        # Would be nice to have a scheme to specify bins without listing them all
        # TODO warn if you chose an observable that requires bins
        if 'bins' in obs_cfg:
            self.scale_bins = np.array((obs_cfg['bins']))
            del obs_cfg['bins']
            self.n_bins = len(self.scale_bins)-1
        else:
            # For things like number density, don't need bins. this is temporary though.
            self.scale_bins = None
            self.n_bins = 1

        # number of repopulations to average together for each cosmo/hod combination
        # default is 1
        n_repops = 1
        if 'n_repops' in obs_cfg:
            n_repops = int(obs_cfg['n_repops'])
            assert n_repops > 1
            del obs_cfg['n_repops']
        self._n_repops = n_repops

        # whether or not to take the log of the observable after calculation
        # certain things, like xi and wp, are easier to emulate in log space
        # will define a tranform function for either case
        log_obs = bool(obs_cfg['log_obs']) if 'log_obs' in obs_cfg else True
        self._transform_func = np.log10 if log_obs else lambda x: x

        if 'log_obs' in obs_cfg:
            del obs_cfg['log_obs']

        # This goes through the motions of getting calc observable.
        # Each cat will have to retrieve their own version though.

        dummy_cat = self.cats[0]
        try:
            # get the function to calculate the observable
            calc_observable = getattr(dummy_cat, 'calc_%s' % self.obs)
        except AttributeError:
            raise AttributeError(
                "Observable %s is not available. Please check if you specified correctly, only\n observables that are calc_<> cat functions are allowed." %self.obs)

        # check to see if there are kwargs for calc_observable
        args = calc_observable.args  # get function args
        kwargs = {}
        # obs_cfg may have args for our function.
        if any(arg in obs_cfg for arg in args):
            kwargs.update({arg: obs_cfg[arg] for arg in args if arg in obs_cfg})
        if n_repops > 1:  # don't do jackknife in this case
            if 'do_jackknife' in obs_cfg and obs_cfg['do_jackknife']:
                warnings.warn('WARNING: Cannot perform jackknife with n_repops>1. Turning off jackknife.')
                if 'do_jackknife' in args:
                    kwargs['do_jackknife'] = False

        self._calc_observable_kwargs = kwargs

    def _get_calc_observable(self, cat):
        """
        Get the calc_observable function for this cat. Ex, will retrieve a reference to calc_xi
        if the observabel is xi. Additionally, if log_obs is true, will wrap in a log10 automatically.
        :param cat:
            The cat object to retrive calc_observabel for.
        :return:
            calc_observable, a function that calculates the observable for cat with no args
        """
        # TODO wanna make sure this works with no args
        _calc_observable = getattr(cat, 'calc_%s' % self.obs)
        if self._calc_observable_kwargs:
            if self.scale_bins is not None:
                return lambda : _calc_observable(self.scale_bins, **self._calc_observable_kwargs)
            else:
                return lambda : _calc_observable(**self._calc_observable_kwargs)
        else:
            if self.scale_bins is not None:
                return lambda : _calc_observable(self.scale_bins)
            else:
                return _calc_observable



    def prep_computation(self, comp_cfg):
        """
        #TODO
        """

        # here i could export enviornment variables that ensure the cores per node is right?
        self.output_fname = comp_cfg['filename']
        overwrite = comp_cfg['overwrite'] if 'overwrite' in comp_cfg else False
        if path.exists(self.output_fname) and not overwrite:
            raise IOError("Overwrite is False or Not Specified, but the file exists. Please change the config.")


    def _get_group_name(self, cosmo_idx, scale_factor):
        """
        Helper function to return a standard group/dataset name based on the indicies
        # TODO
        """
        return "cosmo_no_%02d/a_%.3f"%(cosmo_idx, self._scale_factors[scale_factor])


    def run(self, comm):
        """
        #TODO
        """
        # a list of the indices ofo the cosmology, scale factor, and hod to be computed.
        # will be scattered over all processes

        rank = comm.Get_rank()
        size = comm.Get_size()

        # TODO since these are numpy objects, could be communicated more efficiently
        # since they're small and its only once, I don't think it matters much.
        sendbuf = None
        all_param_idxs = np.array(list(product(xrange(len(self.cats)), xrange(len(self._scale_factors)),
                                    xrange(self._hod_param_vals.shape[0]))))
        n_combos = len(all_param_idxs)
        n_per_node = n_combos/size
        remainder = n_combos%size


        if rank == 0:

            zero_remainder = remainder == 0
            if not zero_remainder:
                n_per_node +=1

            sendbuf = np.empty([size, n_per_node, 3], dtype = 'i')
            
            past_remainder_counter = 0
            # This could be cleaned up to be more readable
            for i in xrange(size):
                if remainder == 0:
                    if not zero_remainder: # initially, after we used it as a counter
                        sendbuf[i, :-1, :] =  all_param_idxs[i*n_per_node - past_remainder_counter:(i+1)*n_per_node-past_remainder_counter-1, :]
                        sendbuf[i,-1, :] = np.array([-1,-1,-1])
                        past_remainder_counter+=1
                    else: #no extra lines
                        sendbuf[i, :, :] =  all_param_idxs[i*n_per_node:(i+1)*n_per_node, :]

                else:
                    sendbuf[i,:,:] = all_param_idxs[i*n_per_node:(i+1)*n_per_node, :]
                    remainder-=1

            print size, n_per_node, sendbuf.shape, all_param_idxs.shape


#        else:
#            all_param_idxs = None

        param_idxs= np.empty([n_per_node, 3], dtype = 'i')
        #all_param_idxs = comm.scatter(all_param_idxs, root=0)
        comm.Scatter(sendbuf, param_idxs, root = 0)


        #print all_param_idxs

        if self.n_bins > 1 :
            output = np.zeros((param_idxs.shape[0], self.n_bins))
            output_cov = np.zeros((param_idxs.shape[0], self.n_bins, self.n_bins))

        else:
            output = np.zeros((param_idxs.shape[0],))
            output_cov = np.zeros((param_idxs.shape[0]))


        last_cosmo_idx, last_scale_factor_idx = -1, -1
        t0 = time()
        for output_idx, (cosmo_idx, scale_factor_idx, hod_idx) in enumerate(param_idxs):
            print 'Rank: %d, Cosmo: %d, Scale_Factor: %d, HOD: %d'%(rank, cosmo_idx, scale_factor_idx, hod_idx)
            print 'Time: %.2f'%(time()- t0)
            print '*'*30

            if any(idx == -1 for idx in [cosmo_idx, scale_factor_idx, hod_idx]):
                continue # skip these placeholders
            if last_cosmo_idx != cosmo_idx or last_scale_factor_idx != scale_factor_idx:
                if last_cosmo_idx != -1:
                    # free some memory
                    last_cat = self.cats[last_cosmo_idx]
                    del last_cat.halocat
                    del last_cat.model

                cat = self.cats[cosmo_idx]

                scale_factor = self._scale_factors[scale_factor_idx]

                cat.load(scale_factor, HOD= self.hod, particles = self._particles,
                         downsample_factor=self._downsample_factor, hod_kwargs= self._hod_kwargs  )

                calc_observable = self._get_calc_observable(cat)

            hod_params = dict(zip(self._hod_param_names, self._hod_param_vals[hod_idx, :]))
            if self._n_repops == 1:
                cat.populate(hod_params)
                # TODO this will fail if you don't jackknife when n_repops is 1
                obs_val, obs_cov = self._transform_func(calc_observable())
            else:  # do several repopulations
                if self.n_bins > 1:
                    obs_repops = np.zeros((self._n_repops, self.n_bins))
                else:
                    obs_repops = np.zeros((self._n_repops,))

                for repop in xrange(self._n_repops):
                    cat.populate(hod_params)
                    obs_repops[repop] = self._transform_func(calc_observable())

                obs_val = np.mean(obs_repops, axis=0)
                obs_cov = np.cov(obs_repops, rowvar=False)
            output[output_idx] = obs_val
            output_cov[output_idx] = obs_cov

            last_cosmo_idx = cosmo_idx
            last_scale_factor_idx = scale_factor_idx

        # TODO, someday, glorious parallel hdf5.
        # the "gather" should at max collect 16mb, so it may not be sooo bad.
        #f = h5py.File(self.output_fname, 'w', driver='mpio', comm=comm)

        all_output = None
        all_output_cov = None
        if rank == 0:
            if self.n_bins == 1:
                all_output = np.empty([size, n_per_node], dtype=output.dtype)
                all_output_cov = np.empty([size, n_per_node], dtype = output_cov.dtype)
            else:
                all_output = np.empty([size, n_per_node, self.n_bins], dtype=output.dtype)
                all_output_cov = np.empty([size, n_per_node, self.n_bins, self.n_bins], output_cov.dtype)
        
        comm.Gather(output, all_output, root=0)
        comm.Gather(output_cov, all_output_cov, root = 0)


        if rank == 0:
            # wrap things up in the final process
            if self.n_bins == 1:
                all_output = all_output.reshape((-1,))
                all_output_cov = all_output_cov.reshape((-1,))
            else:
                all_output = all_output.reshape((-1,self.n_bins))
                all_output_cov = all_output_cov.reshape((-1,self.n_bins, self.n_bins))

            all_cosmo_sf_pairs = np.array(list(product(xrange(len(self.cats)), xrange(len(self._scale_factors)))))

            f = h5py.File(self.output_fname, 'w')
            f.attrs['obs'] = self.obs
            f.attrs['cosmo_param_names'] = self._cosmo_param_names
            f.attrs['cosmo_param_vals'] = self._cosmo_param_vals
            f.attrs['scale_factors'] = self._scale_factors
            f.attrs['hod_param_names'] = self._hod_param_names
            f.attrs['hod_param_vals'] = self._hod_param_vals
            f.attrs['scale_bins'] = self.scale_bins

            for cosmo_sf_pair in all_cosmo_sf_pairs:
                group_name = self._get_group_name(*cosmo_sf_pair)
                grp = f.create_group(group_name) # could rename the above to the group name

                # I could compute this, which would be faster, but this is easier to read.
                hod_idxs = np.where(np.all(all_param_idxs[:, :2] == cosmo_sf_pair, axis = 1))[0]


                grp.create_dataset("obs", data=all_output[hod_idxs], chunks = True, compression = 'gzip')
                grp.create_dataset("cov", data=all_output_cov[hod_idxs], chunks = True, compression = 'gzip')

            f.close()

if __name__ == '__main__':
    from sys import argv
    config_fname = argv[1] # could implement full argparse if i want i guess
    comm = MPI.COMM_WORLD

    trainer = Trainer(config_fname)
    trainer.run(comm)
