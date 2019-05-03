# take the BigFile format data from FastPM and convert it into one nice HDF5 file
from os import path
from glob import glob
from ast import literal_eval
import pandas as pd
import numpy as np
import yaml
import h5py
from nbodykit.source.catalog.file import BigFileCatalog
from halotools.sim_manager import UserSuppliedHaloCatalog

HALO_COLUMNS=['ID', 'Mass', 'Position']#, 'RVdisp', 'Rdisp', 'Vdisp']
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
        elif col == 'Position':
            cat[:idx+1:idx+1+3] = halocat[col]
        else: #TODO this is gonna break for position
            cat[:,idx+1] = halocat[col]

    grp.create_dataset("halos", data = cat, compression="gzip")
    return pmass

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

    pmasses = []
    for box_no, box_dir in enumerate(boxdirs):
        grp = f.create_group("Box_%03d"%box_no)
        scale_factors = glob(path.join(box_dir, 'fastpm_*'))
        for sfdir in scale_factors:
            a = float(sfdir.split('_')[1])
            z = 1.0/a -1.0
            grp2 = grp.create_group("z=%0.3f"%z)
            pmass = create_halo_dset(grp2, sfdir)
            #create_particle_dset(grp, box_dir)

        pmasses.append(pmass)

    f.attrs['pmasses'] = pmasses
    f.close()

def cache_halotools(hdf5_fname, output_dir):

    assert path.exists(hdf5_fname)
    assert path.exists(output_dir)

    f = h5py.File(hdf5_fname, 'r')
    pmasses = np.array(f.attrs['pmasses'])
    cfg = literal_eval(f.attrs['cfg_info'])
    Lbox = cfg['boxsize']

    for pm, box_key in zip(pmasses, f.keys()):
        for z_key in f[box_key].keys():
            z = float(z_key.split('_')[1])
            print box_key, z_key

            data = f[box_key][z_key]['halos'].value
            # assuming halo_columns
            # TODO do this better
            halo_id = data[:,0]
            halo_mass = data[:,1]
            halo_x, halo_y, halo_z = data[:,2], data[:,3], data[:,4]

            halocat = UserSuppliedHaloCatalog(redshift = z, Lbox=Lbox, pmass = pm,
                                              halo_id = halo_id, halo_mass = halo_mass,
                                              halo_x = halo_x, halo_y = halo_y, halo_z=halo_z)

            version_name = box_key+'_'+z_key
            halocat.add_halocat_to_cache(fname = path.join(output_dir, version_name, '.hdf5'),
                                         simname = 'FastPM', halo_finder = 'FOF',
                                         version_name=version_name, overwrite=True)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert the sim outputs to one hdf5 file.')
    parser.add_argument('config_fname', help='Filename of the configfile.')
    parser.add_argument('output_fname', help= "Filename of output hdf5 file")
    args = parser.parse_args()
    config_fname = args.config_fname
    output_fname = args.output_fname

    make_hdf5_file(config_fname, output_fname)
