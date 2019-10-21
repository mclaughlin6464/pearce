from os import path, mkdir
from subprocess import call
from time import time
import numpy as np
import pandas as pd
import yaml
from nbodykit.cosmology import Cosmology
from nbodykit.cosmology.power import LinearPower, ZeldovichPower

# I could have this accept kwargs,
# and have main
def submit_sims(lhc_fname, output_dir, powerspectrum, cosmic_var, initial_z, final_z,\
                npart, boxsize, ncores, sim_time):

    assert path.isdir(output_dir), "Invalid output directory %s"%output_dir

    lhc = pd.read_csv(lhc_fname, sep=' ', index_col=None)
    param_names = list(lhc)

    k = np.logspace(-4, 2, 1000)
    # others?
    assert powerspectrum.lower() in {'linear', 'zeldovich'}, "Invalid initial power spectrum"

    if powerspectrum.lower() == 'linear':
        power = LinearPower
    else:
        power = ZeldovichPower

    cosmic_var = str(cosmic_var)
    assert cosmic_var.lower() in {'true', 'false'}

    initial_a, final_a = 1./(1+initial_z), 1./(1+final_z)

    for idx, row in lhc.iterrows():
        # TODO what to do if already exists? likely will throw some kind of error
        print idx
        boxdir = path.join(output_dir, "Box_%03d/"%idx)
        if not path.isdir(boxdir):
            mkdir(boxdir)

        cosmo = make_cosmo(param_names, row)
        p = power(cosmo, 0.0)(k) #only works for z=0

        np.savetxt(path.join(boxdir,'powerspec.txt'), np.array((k,p)).T)

        with open(path.join(boxdir, 'nbodykit.lua'), 'w') as f:
            f.write(sim_config_template.format(nc = npart, boxsize = boxsize,
                                              initial_a = initial_a,
                                              final_a = final_a,
                                              final_z = final_z,
                                              cosmic_var = cosmic_var,
                                              omega_m = cosmo.Om0,
                                              h = cosmo.h,
                                              seed = int(time())%1000,
                                              boxdir = boxdir))

        # now, write submission file and submit

        with open(path.join(boxdir, 'submit_sim.sbatch'), 'w') as f:
            f.write(submission_file_template.format(jobname = 'Box_%03d'%idx,
                                                    boxdir = boxdir,
                                                    time = sim_time*60, #TODO in config
                                                    ntasks = ncores))

        call("sbatch {boxdir}submit_sim.sbatch".format(boxdir=boxdir), shell=True)

def make_cosmo(param_names, param_vals):
    """
    Frequently, the parameters of our input will not align with what we need.
    So, we convert them to the appropriate format.
    :param param_names:
    :param param_vals:
    :return:
    """
    param_dict = dict([(pn, param_vals[pn]) for pn in param_names] )
    # TODO it'd be nice if this could be somehow generalized.
    param_dict['N_ncdm'] = 3.0
    param_dict['N_ur'] = param_dict['Neff']-3.0
    del param_dict['Neff']

    #param_dict['h'] = param_dict['H0']/100
    #del param_dict['H0']
    param_dict['h'] = param_dict['h0']
    del param_dict['h0']

    #param_dict['w0_fld'] = param_dict['w']
    w = param_dict['w']
    del param_dict['w']
    # TODO subtract neutrinos from omega_m too!

    param_dict['Omega_cdm'] = param_dict['Omega_m'] - param_dict['Omega_b']
    del param_dict['Omega_b']
    del param_dict['Omega_m']

    param_dict['Omega_ncdm'] = [param_dict['Omeganuh2']/(param_dict['h']**2), 0.0, 0.0]
    #param_dict['m_ncdm']=None
    #param_dict['omch2'] = (param_dict['Omega_m'] - param_dict['Omega_b'] - param_dict['Omega0_ncdm_tot'])*(param_dict['h']**2)
    del param_dict['Omeganuh2']

    #param_dict['A_s'] = 10**(np.log10(np.exp(param_dict['ln_1e10_A_s']))-10.0)
    param_dict['A_s'] = 10**(-10)*np.exp(param_dict['ln(10^{10}A_s)'])
    del param_dict['ln(10^{10}A_s)']

    #print param_dict
    # this is seriously what i have to do here.
    C= Cosmology()
    C2 = C.from_dict(param_dict)
    C3 = C2.clone(w0_fld = w)
    return C3#Cosmology(**param_dict)



sim_config_template = """
nc = {nc:d}
boxsize  = {boxsize:.1f}

time_step = linspace({initial_a:.3f}, 1.0, 30)

output_redshifts = {{{final_z:.2f}, 0.0}} --redshift of output

omega_m = {omega_m: f}
h = {h:f}

read_powerspectrum = "{boxdir}powerspec.txt"
remove_cosmic_variance = {cosmic_var} 
random_seed = {seed:d}

pm_nc_factor = 2
np_alloc_factor = 4.0

fof_nmin = 30

write_snapshot = "{boxdir}fastpm"
write_nonlineark= "{boxdir}fastpm"
write_fof = "{boxdir}fastpm"

-- 1d power spectrum (raw), without shotnoise correction
write_powerspectrum = "{boxdir}powerspec-debug"
"""

submission_file_template = """#!/bin/bash
#SBATCH -p regular
#SBATCH --time={time:d}:00
##SBATCH --qos regular
#SBATCH -A cosmosim
#SBATCH --mail-type=ALL
#SBATCH --mail-user=swmclau2@stanford.edu
#SBATCH -N {ntasks:d} 
#SBATCH --ntasks={ntasks:d}
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=MaxMemPerCPU
#SBATCH -J {jobname} 
#SBATCH --output={boxdir}{jobname}.out
#SBATCH --error={boxdir}{jobname}.err 


srun /global/u2/s/swmclau2/Git/fastpm/src/fastpm {boxdir}nbodykit.lua
"""
#may need to be an mpirun?


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Submit an ensemble of simulations.')
    parser.add_argument('config_fname', help = 'Filename of the configfile.')
    # TODO here is where i would add an argument to parse what sims to run
    # TODO add arg to do halos, or should that just be a separate file.
    args = parser.parse_args()
    config_fname = args.config_fname
    # NOTE could use a vars(args) to turn into a dictionary.
    # currently not doing that if some of those end up not being arguments for submit_sims
    try:
        assert path.isfile(config_fname)
    except AssertionError:
        raise AssertionError("%s is not a valid filename." % config_fname)

    with open(config_fname, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    try:
        submit_sims(**cfg)
    except TypeError:
        raise TypeError("Not all require arguments specified in yaml config.")
