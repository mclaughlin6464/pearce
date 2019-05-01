# take the BigFile format data from FastPM and convert it into one nice HDF5 file
from os import path
from glob import glob
import pandas as pd
import numpy as np
import yaml
import h5py
from nbodykit.source.catalog.file import BigFileCatalog

HALO_COLUMNS=['ID', 'Mass', 'Position', 'RVdisp', 'Rdisp', 'Vdisp']
PARTICLE_COLUMNS=[]
MASS_UNIT = 1e10 # can get this from file

def create_halo_dset(grp, dirname, columns=HALO_COLUMNS):

    halocat = BigFileCatalog(dirname, dataset='LL-0.200') # TODO make modifiable

    first_col = np.array(halocat[columns[0]])

    cat = np.zeros((first_col.shape[0], len(columns)))

    cat[:, 0] = first_col

    with open(path.join(dirname, 'attr-v2'), 'r') as f:
        for line in f:
            pmass = float(line.split()[-2])*MASS_UNIT
            break

    for idx, col in columns[1:]:

        if col == 'Mass':
           cat[:, idx+1] = halocat['Length']*pmass
        else:
            cat[:,idx+1] = halocat[col]

    grp.create_dataset("halos", data = cat, compression="gzip")

def create_particle_dset(grp, dirname, columsn=PARTICLE_COLUMNS):
    raise NotImplementedError

def make_hdf5_file(config_fname, fname):

    assert path.exists(config_fname)
    with open(config_fname, 'r') as yamlfile:
        cfg = yaml.load(yamlfile)

    dirname = cfg['output_dir']

    lhc = pd.read_csv(cfg['lhc_fname'], sep=' ', index_col=None)
    lhc_pnames = list(lhc)
    lhc_vals = lhc.to_numpy()

    f = h5py.File(fname, 'w')

    f.attrs['lhc_pnames'] = lhc_pnames
    f.attrs['lhc_vals'] = lhc_vals
    f.attrs['halo_pnames'] = HALO_COLUMNS
    #f.attrs['particle_pnames'] = PARTICLE_COLUMNS
    f.attrs['cfg_info'] = str(cfg)

    boxdirs = sorted(glob(path.join(dirname, "Box_*/")))

    for box_no, box_dir in enumerate(boxdirs):
        grp = f.create_group("Box_%03d"%box_no)
        create_halo_dset(grp, box_dir)
        #create_particle_dset(grp, box_dir)

    f.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert the sim outputs to one hdf5 file.')
    parser.add_argument('config_fname', help='Filename of the configfile.')
    parser.add_argument('output_fname', help= "Filename of output hdf5 file")
    args = parser.parse_args()
    config_fname = args.config_fname
    output_fname = args.output_fname

    make_hdf5_file(config_fname, output_fname)
