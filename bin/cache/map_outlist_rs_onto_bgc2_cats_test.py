from os.path import join
from csv import DictReader, DictWriter

fieldnames = '#ID DescID M200b Vmax Vrms R200b Rs Np X Y Z VX VY VZ Parent_ID'.split(' ')

missed_keys = dict()

#for cosmo_idx in xrange(40):
for cosmo_idx in xrange(7):
    for realization_idx in xrange(5):
        #halodir = '/home/users/swmclau2/scratch/NewTrainingBoxes/Box%03d/halos/m200b/'%cosmo_idx
        #halodir = '/home/users/swmclau2/scratch/TrainingBoxes/Box000/halos/m200b/'
        halodir = '/home/users/swmclau2/scratch/NewTrainingBoxes/TestBox%03d-%03d/halos/m200b/'%(cosmo_idx, realization_idx)
        for snapshot_idx in xrange(10):
            outlist_fname = join(halodir, "out_%d.list"%snapshot_idx)
            bgc2list_fname = join(halodir, "outbgc2_%d.list"%snapshot_idx)
            bgc2list_fname2 = join(halodir, "outbgc2_rs_%d.list"%snapshot_idx)
                    
            print cosmo_idx, realization_idx, snapshot_idx

            outlist_rs = dict()
            with open(outlist_fname) as csvfile:
                reader = DictReader(filter(lambda row: row[0]!='#' or row[:3]=='#ID', csvfile), delimiter = ' ')
                for row in reader:
                    outlist_rs[row['#ID']] = row['Rs']

            with open(bgc2list_fname) as oldfile, open(bgc2list_fname2, 'w') as newfile:
                reader = DictReader(filter(lambda row: row[0]!='#' or row[:3]=='#ID', oldfile), delimiter = ' ')
                writer = DictWriter(newfile, fieldnames, delimiter = ' ')
                writer.writeheader()
                print_first = True
                for row in reader:
                    if print_first:
                        print row
                        print_first = False
                    try:
                        row['Rs'] = outlist_rs[row['#ID']]
                    except KeyError:
                        missed_keys[(cosmo_idx, realization_idx)] = row['#ID']
                        #continue
                                            
                    writer.writerow(row)

            print len(missed_keys)
