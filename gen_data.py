import os
import shutil
import argparse
import numpy as np

# local imports
from support_modules import extract_particles
from support_modules import extract_star_meta

def gen_data():
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Extract good and bad particles and split into test/validation and training datasets.')
    parser.add_argument('-p', '--projpath', type=str, help='path for project', required=True)
    parser.add_argument('-g', '--stargood', type=str, help='input star file with good particles', required=True)
    parser.add_argument('-b', '--starbad', type=str, help='input star file with bad particles', required=True)
    parser.add_argument('-c', '--cleardata', action='store_true', help='clear training/test data before extracting')
    args = parser.parse_args()
    
    # create directory structure
    work_dir = args.projpath
    if work_dir.endswith('/'):
        work_dir = work_dir.rstrip('/')
    
    data_dir = work_dir + '/ClassBin'
    train_dir = work_dir + '/ClassBin/train'
    test_dir = work_dir + '/ClassBin/test'
    if os.path.exists(data_dir) == False:
        os.mkdir(data_dir)

    # extract box size, box apix, and original image apix
    meta_good = extract_star_meta.extract(args.stargood)
    meta_bad = extract_star_meta.extract(args.starbad)
    
    if meta_good != meta_bad:
        print("Headers for good and bad particle star files do not match. Exiting.")
        exit()
    else:
        print("\nMetadata from STAR files:\nImage apix: " + meta_good[2] + "\nBox size: " + meta_good[0] + "\nBox apix: " + meta_good[1]  + "\n")
        box = int(meta_good[0])
        b_apix = float(meta_good[1])
        i_apix = float(meta_good[2])
        
    # option to clear particle cache and re-extract particles
    if (args.cleardata == True):
        if ('train' in os.listdir(data_dir) or 'test' in os.listdir(data_dir)):
            print('Clearing particle cache...', end="")
            try:
                shutil.rmtree(train_dir)
            except FileNotFoundError:
                pass
            try:
                shutil.rmtree(test_dir)
            except FileNotFoundError:
                pass

            print('done.\n')
        else:
            print('Cannot clear particle cache because no particles found. Exiting.')
            exit()
        
    # create directories to store training/test data
    if ('train' in os.listdir(data_dir) or 'test' in os.listdir(data_dir)):
        print('Particle extraction halted because existing particles found. Re-run with -c flag to clear existing particles. Exiting.\n')
        exit()
    else:
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        good_train_dir = train_dir + '/good'
        os.mkdir(good_train_dir)

        bad_train_dir = train_dir + '/bad'
        os.mkdir(bad_train_dir)

        good_test_dir = test_dir + '/good'
        os.mkdir(good_test_dir)

        bad_test_dir = test_dir + '/bad'
        os.mkdir(bad_test_dir)

        # extract good and bad particle data
        extract_particles.extract(args.projpath, args.stargood, good_train_dir, good_test_dir, 'good')
        extract_particles.extract(args.projpath, args.starbad, bad_train_dir, bad_test_dir, 'bad')
        
        print('Particle extraction complete. Training and test datasets have been created.\n')

if __name__ == "__main__":
   gen_data()