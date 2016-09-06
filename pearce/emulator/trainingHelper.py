#!/bin/bash
'''Training helper. Is called from training data as main, to send to cluster.'''

from os import path
import sys
import numpy as np
#from trainingData import PARAMS, GLOBAL_FILENAME
#from ioHelpers import global_file_reader
#from ..mocks.kittens import cat_dict
#Not happy about this, look into a change.

sys.path.append('..')
from pearce.emulator.trainingData import PARAMS, GLOBAL_FILENAME
from pearce.emulator.ioHelpers import global_file_reader
from pearce.mocks.kittens import cat_dict

# TODO to ioHelpers?
def load_training_params(param_file):
    '''
    Load the parameters to calculate. Will load bins, parameters, cosmology info, output directory, and a unique id!
    :param param_file:
        The file where the parameters are stored. The output directory is the same directory as the paramfile, and
        the radial bins are stored in the same directory as well.
    :return: rbins, parameters, cosmology info, output directory, and job_id.
    '''
    hod_params = np.loadtxt(param_file)
    dirname = path.dirname(param_file)
    rbins, cosmo_params, _ = global_file_reader(path.join(dirname, GLOBAL_FILENAME))

    job_id = int(param_file.split('.')[0][-3:]) #last 3 digits of paramfile is a unique id.

    return hod_params, rbins, cosmo_params, dirname, job_id

def calc_training_points(hod_params, rbins, cosmo_params, dirname, job_id):
    '''
    given an array of hod parameters (and a few other things) populate a catalog and calculate xi at those points.
    :param hod_params:
        An array of hod parameter points. Each row represents a "point" in HOD space. This function iterates over those
        points anc calculates xi at each of them.
    :param rbins:
        Radial bins for the xi calculation.
    :param cosmo_params:
        cosmology information, used for loading a catalog. Required is simname, scale_factor, and anything required to load
        a specific catalog (Lbox, npart, etc.)
    :param dirname:
        The directory to write the outputs of this function.
    :param job_id:
        A unique identifier of this call, so saves from this function will not overwrite parallel calls to this function.
    :return: None
    '''
    cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!
    #Could add a **kwargs to load to shorten this.
    #That makes things a little messier though IMO
    # TODO tol?
    if 'HOD' in cosmo_params:
        cat.load(cosmo_params['scale_factor'], cosmo_params['HOD'])
    else:
        cat.load(cosmo_params['scale_factor'])

    for id, hod in enumerate(hod_params):
        #construct a dictionary for the parameters
        # Could store the param ordering in the file so it doesn't need to be imported.
        hod_dict = {p.name: val for p, val in zip(PARAMS, hod)}
        #Populating the same cat over and over is faster than making a new one over and over!
        cat.populate(hod_dict)
        #TODO pass in do_jackknife, use_corrfunc?
        xi, xi_cov = cat.calc_xi(rbins)
        # Consider storing them all and writing them all at once. May be faster.
        # Writing the hod params as a header. Could possibly recover them from the same file I used to read these in.
        # However, I think storing them in teh same place is easier.
        header_start = ['Cosmology: %s' % cosmo_params['simname'], 'Params for HOD:']
        header_start.extend('%s:%.3f' % (key, val) for key, val in hod_dict.iteritems())
        header = '\n'.join(header_start)
        np.savetxt(path.join(dirname, 'xi_job%03d_HOD%03d.npy'%(job_id, id) ), xi, header=header)
        np.savetxt(path.join(dirname, 'cov_job%03d_HOD%03d.npy'%(job_id, id) ), xi_cov, header=header)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Helper function called as main from "trainingData.py." Reads in \
                                                collections of HOD params and calculates their correlation functions.')
    parser.add_argument('param_file', type = str, help='File where the vector of HOD params are stored.')

    args = vars(parser.parse_args())

    hod_params, rbins, cosmo_params, dirname, job_id = load_training_params(args['param_file'])
    calc_training_points(hod_params, rbins, cosmo_params, dirname, job_id)
