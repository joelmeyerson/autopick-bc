#!/usr/bin/env python3

import os
import shutil
import argparse
import sys
import re
import pathlib
import math
import numpy as np
import mrcfile
from PIL import Image
import matplotlib.pyplot as plt
from progress.bar import Bar

# TF imports
import tensorflow as tf
from tensorflow import keras

# disable GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def gen_picks():
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Use model to classify extracted images as good (true particles) or bad (empty area, ice contamination, gold/carbon film).')
    parser.add_argument('-p', '--projpath', type=str, help='path for project', required=True)
    parser.add_argument('-i', '--imagestar', type=str, help='particle star file containing extracted grid particles for classification using model', required=True)
    args = parser.parse_args()
    
    # format project path 
    work_dir = args.projpath
    if work_dir.endswith('/'):
        work_dir = work_dir.rstrip('/')
    
    # check that sub-directory exists
    data_dir = work_dir + '/ClassBin'
    if os.path.exists(data_dir) == False:
        print("No ClassBin directory found.")
        exit()
    
    # check that model exists
    if os.path.exists(data_dir + '/model.h5') == False:
        print("No model file (h5 format) found.")
        exit()

    # load model
    try:
        model = tf.keras.models.load_model(data_dir + '/model.h5')
    except:
        print("Error loading model. Exiting")
        exit()

    # from star file extract x coord, y coord, particle stack name, particle stack index for each particle
    data_cols = star_columns(args.imagestar)
    data = star_data(args.imagestar)
    
    # create good and bad star file
    star_good = star_head(args.imagestar)
    star_bad = star_head(args.imagestar)
    
    # get index to reference the mrc slice@stack in data
    mrc_stack_idx = data_cols.index('_rlnImageName')
    
    # get box size by reading in a particle stack
    mrc_path = data[0][mrc_stack_idx].split("@")[1] # this will get the first meta line in the star file, and then use idx to get the MRC stack path
    mrc = mrcfile.open(mrc_path, mode=u'r', permissive=False, header_only=False)
    box = len(mrc.data[0,:,:][0])
    mrc.close()

    # setup for iterating through star meta (data) to get batches
    par_cnt = len(data)
    batch_size = 32
    batch_num = math.floor(par_cnt/batch_size) # total number of batches (not counting the modulo)
    batch_fragment_size = par_cnt%batch_size # get number of samples in the final batch (modulo)

    if batch_fragment_size > 0: # if final batch has fewer than 32 samples, make sure it is included in the batch count
        batch_num += 1

    batch_star = [] # for each batch store the lines for particle star file (32 entries)
    
    # create progress bar
    bar = Bar('\nProcessing ' + str(par_cnt) + ' samples in ' + str(batch_num) + ' batches:', fill='#', suffix='%(percent)d%%', max=batch_num)

    # each round of loop will predict on a batch
    for batch_cnt in range(0,batch_num):
        # set metadata range for current batch
        if ((batch_cnt == batch_num-1) and (batch_fragment_size > 0)):
            start = batch_size*(batch_cnt)
            end = start + batch_fragment_size - 1

        else:
            start = batch_size*(batch_cnt)
            end = start + batch_size - 1

        batch_star = data[start:end + 1] # xcoord, ycoord, particle name, and slice for all particles in a batch
        # NOTE: python list indexing is not inclusive, so need to extend the end point by one in above statement

        # load mrc batch into array
        batch_img = np.zeros(shape=(len(batch_star),box,box,1)) # for each batch store the image batch (32 entries, or fewer for final batch)
        loader_cnt = 0

        for sample in batch_star: # each "sample" is a row in the star file

            mrc_slice = int(sample[mrc_stack_idx].split("@")[0].lstrip("0")) - 1
            mrc_stack = sample[mrc_stack_idx].split("@")[1]
            mrc = mrcfile.open(mrc_stack, mode=u'r', permissive=False, header_only=False)

            img = np.flip(mrc.data[:, :])
            img = np.flip(mrc.data[mrc_slice, :, :], axis=0)
            img = img + abs(img.min()) # make all 32 bit floating point pixel values >= 0
            img /= img.max() # normalize all pixels between 0 and 1
            img *= 255 # normalize all pixels between 0 and 255

            batch_img[loader_cnt,:,:,0] = img
            loader_cnt += 1

        # predict on batch
        batch_pred = model.predict(batch_img)
        batch_pred = abs(batch_pred.round())
        batch_pred = np.array(batch_pred[:,0]) # convert tuple to array

        # update star arrays
        pos_idx = np.argwhere(batch_pred==1).flatten()
        for p in pos_idx:
            star_good.append(batch_star[p])

        neg_idx = np.argwhere(batch_pred==0).flatten()
        for n in neg_idx:
            star_bad.append(batch_star[n])

        bar.next()

    # write true and false particle star files to disk
    bar.finish()
    star_fname_good = star_write(args.imagestar, star_good, "true")
    star_fname_bad = star_write(args.imagestar, star_bad, "false")

    print("\nResults written to the star files:\n")
    print(star_fname_good)
    print(star_fname_bad)
    print("")

# extract header from particles.star and store in list
def star_head(star):
    
    header_array = []
    with open(star, "r") as openfile:
        for line in openfile:
            
            # exit when header block is over
            if re.search(r'@', line):
                return header_array
            
            else:
                header_array.append(line)

# extract the column labels and their column number and store in list
def star_columns(star):
    
    in_data_block = False
    
    cols = []
    with open(star, "r") as openfile:
        for line in openfile:

            # exit when header block is over
            if re.search(r'@', line):
                return cols
            
            # look for data_particles block
            if (in_data_block == False and re.search(r'data_particles', line)):
                in_data_block = True
            
            if (in_data_block == True and '_r' in line):
                tag = str(line.split()[0]) # e.g. "_rlnCoordinateX"
                #num = int(str(line.split()[1]).strip("#")) # e.g. "1"
                cols.append(tag)

# extract data in star file (all lines with @ symbol)
def star_data(star):
    
    data = []
    with open(star, "r") as openfile:
        for line in openfile:

            # only and all data entries have @ character
            if re.search(r'@', line):
                data.append(line.split())
    
    return data

# write results to true and false star files
def star_write(star_file, star_array, tf):
    
    fname = os.path.splitext(str(star_file))[0] + "_" + tf + ".star"
    with open(fname, "w") as starfile:
        
        for line in star_array:
            if re.search(r'@', '\t'.join(line)): # this is a data entry which needs to be converted from array to tab-delimited string
                starfile.writelines("%s\n" % '\t'.join(line))
            else:
                starfile.writelines("%s" % line) # this is part of the header where each line is already a string
    
    return fname
       
if __name__ == "__main__":
   gen_picks()