import os
import shutil
import argparse
import sys
import re
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

    model = tf.keras.models.load_model(data_dir + '/model.h5')
    # load model
    try:
        model = tf.keras.models.load_model(data_dir + '/model.h5')
    except:
        print("Error loading model. Exiting")
        exit()

    
    # from star file extract x coord, y coord, particle stack name, particle stack index for each particle
    meta = []
    with open(args.imagestar, "r") as openfile:
        for line in openfile:

            # get column for x coordinate
            if re.search(r'_rlnCoordinateX', line):
                x_col = int(str(line.split()[1]).strip("#"))
                
            # get column for y coordinate
            if re.search(r'_rlnCoordinateY', line):
                y_col = int(str(line.split()[1]).strip("#"))
                
            # get column for particle name 
            if re.search(r'_rlnImageName', line):
                par_col = int(str(line.split()[1]).strip("#"))

            # use keywords to find lines with micrograph names
            if re.search(r'(mrc | mrcs | tif | tiff)', line):
                
                x = float(line.split()[x_col-1])
                y = float(line.split()[y_col-1])
                par = str(line.split()[par_col-1])
                par_stack = str(par.split('@')[1])
                par_idx = int(par.split('@')[0].lstrip('0')) - 1
                meta.append([x, y, par_stack, par_idx])
    
    # create good and bad star file
    star_good = star_head()
    star_bad = star_head()
    
    # get box size by reading in a particle stack
    mrc_path = work_dir + '/' + meta[0][2]
    mrc = mrcfile.open(mrc_path, mode=u'r', permissive=False, header_only=False)
    box = len(mrc.data[0,:,:][0])
    mrc.close()
    
    # setup for iterating through star meta to get batches
    par_cnt = len(meta)
    batch_size = 32
    batch_num = math.floor(par_cnt/batch_size) # total number of batches (not counting the modulo)
    batch_fragment_size = par_cnt%batch_size # get number of samples in the final batch (modulo)

    if batch_fragment_size > 0: # if final batch has fewer than 32 samples, make sure it is included in the batch count
        batch_num += 1

    batch_star = [] # for each batch store the lines for particle star file (32 entries)
    
    # each round of loop will predict on a batch
    for batch_cnt in range(0,batch_num): 
        # set metadata range for current batch
        if ((batch_cnt == batch_num-1) and (batch_fragment_size > 0)):
            start = batch_size*(batch_cnt)
            end = start + batch_fragment_size

        else:
            start = batch_size*(batch_cnt)
            end = start + batch_size

        batch_star = meta[start:end] # xcoord, ycoord, particle name, and slice for all particles in a batch

        # load mrc batch into array
        batch_img = np.zeros(shape=(len(batch_star),box,box,1)) # for each batch store the image batch (32 entries, or fewer for final batch)
        loader_cnt = 0
        for sample in batch_star:

            mrc_stack = work_dir + '/' + sample[2]
            mrc_slice = sample[3]
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
        pos_idx = np.argwhere(batch_pred==1)
        for idx in pos_idx:
            pass
            #star_good.append(str(format(meta[0], '.1f').rjust(10)) + ' ' + str(format(meta[1], '.1f').rjust(10)) + '            2   -999.00000   -999.00000')
    
def star_head():
    
    star = []
    star.append('')
    star.append('# version 30001')
    star.append('')
    star.append('data_')
    star.append('')
    star.append('loop_')
    star.append('_rlnCoordinateX #1')
    star.append('_rlnCoordinateY #2')
    star.append('_rlnClassNumber #3')
    star.append('_rlnAnglePsi #4')
    star.append('_rlnAutopickFigureOfMerit #5')
    
    return star
    
def star_update(star, par):
    star.append(par)
    return star
    
# def star_write():
#     return
    
if __name__ == "__main__":
   gen_picks()