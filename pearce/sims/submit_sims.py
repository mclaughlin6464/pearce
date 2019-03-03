from os import path, mkdir
from subprocess import call
from time import time
import numpy as np
import pandas as pd
import yaml
from nbodykit.cosmology import Cosmology
from nbodykit.cosmology.power import EHPower, ZeldovichPower

# I could have this accept kwargs,
# and have main
def submit_sims(lhc_fname, output_dir, powerspectrum, initial_z, final_z,\
                npart, boxsize, ncores, sim_time):

    assert path.isdir(output_dir), "Invalid output directory %s"%output_dir

    lhc = pd.read_csv(lhc_fname, sep=' ', index_col=None)
    param_names = list(lhc)

    k = np.logspace(-4, 2, 1000)
    # others?
    assert powerspectrum.lower() in {'linear', 'zeldovich'}, "Invalid initial power specturm"

    if powerspectrum.lower() == 'linear':
        power = EHPower
    else:
        power = ZeldovichPower

    initial_a, final_a = 1./(1+initial_z), 1./(1+final_z)

    for idx, row in lhc.iterrow():
        # TODO what to do if already exists? likely will throw some kind of error
        boxdir = path.join(output_dir, "Box_%03d/"%idx)
        mkdir(boxdir)

        # TODO parse Jeremy's header, or adjust it to be compliant here
        param_dict = dict([(pn, row[pn]) for pn in param_names] )
        cosmo = Cosmology(**param_dict)
        # TODO i wanted to do this with fixed amplitudes, not sure how
        p = power(cosmo, initial_z)

        np.savetxt(path.join(boxdir,'powerspec.txt'), np.array((k,p)).T)

        with open(path.join(boxdir, 'nbodykit.lua')) as f:
           f.write(sim_config_template.format(nc = npart, boxsize = boxsize,
                                              initial_a = initial_a,
                                              final_a = final_a,
                                              final_z = final_z,
                                              omega_m = cosmo.Om0,
                                              h = cosmo.h,
                                              seed = int(time())%1000,
                                              boxdir = boxdir))

        # now, write submission file and submit

        with open(path.join(boxdir, 'submit_sim.sbatch')) as f:
            f.write(submission_file_template.format(jobname = 'Box_%03d'%idx,
                                                    boxdir = boxdir,
                                                    time = sim_time*60, #TODO in config
                                                    ntasks = ncores))

        call("sbatch {boxdir}submit_sim.sbatch".format(boxdir), shell=True)

sim_config_template = """
nc = {nc:d}
boxsize  = {boxsize:.1f}

time_step = linspace({initial_a:.1f}, {final_a:.1f}, 10)

output_redshifts = {{final_z:.2f}} --redshift of output

omega_m = {omega_m: f}
h = {h:f}

read_powerspectrum = "{boxdir}powerspec.txt"
random_seed = {seed:d}

pm_nc_factor = 2
np_alloc_factor = 4.0

write_snapshot = "{boxdir}fastpm"
write_nonlineark= "{boxdir}fastmpm"

-- 1d power spectrum (raw), without shotnoise correction
write_powerspectrum = "{boxdir}powerspec-debug"
"""

submission_file_template = """#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH -p iric
#SBATCH --output={boxdir}{jobname}.out
#SBATCH --error={boxdir}{jobname}.err 
#SBATCH --time={time:d}:00
#SBATCH --ntasks={ntasks:d}

srun /home/users/swmclau2/Git/fastpm/src/fastpm {boxdir}nbodykit.lua
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