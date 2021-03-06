#!/usr/bin/env python3

import os
import sys
import re
import numpy as np
import mrcfile
from PIL import Image, ImageEnhance, ImageDraw, ImageOps
import argparse

CONTRAST_FACTOR = 4
BRIGHTNESS_FACTOR= 100
CIRCLE_LINE_WIDTH = 10

def show_picks():

    # parse arguments
    parser = argparse.ArgumentParser(description='Generate overlay of coordinates on image.')
    parser.add_argument('-p', '--projpath', type=str, help='path for project', required=True)
    parser.add_argument('-m', '--image', type=str, help='image file in mrc format', required=True)
    parser.add_argument('-s', '--star', type=str, help='input star file coordinates to visualize', required=True)
    parser.add_argument('-z', '--binfactor', type=float, help='binning factor used in particle extraction (e.g. 256-->64 is 4)', required=False)
    #parser.add_argument('-o', '--output', type=str, help='file name with png extension', required=False)
    args = parser.parse_args()
    
    mrc = mrcfile.open(args.image, mode=u'r', permissive=False, header_only=False)

    # check that mrc files are flat (not stacks)
    if mrc.data.ndim > 2:
        print("Images must be flat (Z = 1). Cannot process multi-layer images.")
        exit()
        
    # format project path 
    work_dir = args.projpath
    if work_dir.endswith('/'):
        work_dir = work_dir.rstrip('/')
    
    # make sub-directory if it doesn't exist
    data_dir = work_dir + '/ClassBin'
    if os.path.exists(data_dir) == False:
        os.mkdir(data_dir)
    
    # make png sub-directory if it doesn't exist
    png_dir = data_dir + '/pngs'
    if os.path.exists(png_dir) == False:
        os.mkdir(png_dir)
    
    # set bin factor
    bin_fac = 1
    if type(args.binfactor) == float:
        bin_fac = args.binfactor
    
    # process image
    img_array = np.flip(mrc.data, axis=0)
    #mrc.close()
    img_array = img_array + abs(img_array.min()) # make all 32 bit floating point pixel values >= 0
    img_array /= img_array.max() # normalize all pixels between 0 and 1
    img_array *= 255 # normalize all pixels between 0 and 255

    img_base_name = args.image.split('/')[-1].split('.')[0] # get file base name
    image = Image.fromarray(img_array).convert("L")
    image = ImageEnhance.Contrast(image).enhance(CONTRAST_FACTOR)
    image = ImageEnhance.Contrast(image).enhance(BRIGHTNESS_FACTOR)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    #image = ImageOps.mirror(image)
    
    # initialize image for drawing coords
    draw = ImageDraw.Draw(image)   
    
    box = get_box(args.star) * bin_fac
    #box = 216 # for testing
    coords = get_coords(args.image, args.star)
    
    # draw coordinates
    for el in coords:
        x = el[0]
        y = el[1]
        
        #draw.ellipse([x-box/4, y-box/4, x+box/4, y+box/4], fill=None, outline='black', width=CIRCLE_LINE_WIDTH) # make circle 1/2 box size
        draw.ellipse([x-box/2, y-box/2, x+box/2, y+box/2], fill=None, outline='black', width=CIRCLE_LINE_WIDTH) # make circle full box size
        
    # write png
    mrc_base = str(args.image.split('/')[-1].split('.')[0])
    star_ext = str(args.star.split('/')[-1].split('.')[0].split('_')[-1])
    fname = png_dir + '/' + mrc_base + '_' + star_ext + '.png'
    image.save(fname)
    print("\nMicrograph with coordinates overlay PNG saved to: " + fname + "\n")

# get box size from star file
def get_box(star):
    
    in_data_optics_block = False
    box_idx = 0
    
    with open(star, "r") as openfile:
        for line in openfile:
        
            # look for data_optics block
            if (in_data_optics_block == False and re.search(r'data_optics', line)):
                in_data_optics_block = True
        
            # find the header with box size
            if re.search(r'_rlnImageSize', line):
                box_idx = int(str(line.split()[1]).strip("#"))
            if (in_data_optics_block == True and box_idx > 0):        
                if '_r' in line:
                    pass
                    
                else:    
                    box = int(str(line.split()[box_idx-1]))
                    return box
            
            if re.search(r'@', line):
                print("No box size found in star file")
                exit()
            
def get_coords(mrc, star):
    
    # store lines containing mrc file
    coords = []
    
    # mrc file base name
    mrc_base = str(mrc.split('/')[-1].split('.')[0])
    
    # initialize vars to hold index for x and y coordinate columns
    x_idx = 0
    y_idx = 0
    x = 0.0
    y = 0.0
    in_loop = False
    with open(star, "r") as openfile:
        for line in openfile:
            
            # in loop_ section?
            if (re.search(r'loop_', line) and in_loop == False):
                in_loop = True
            
            # find the header entry which contain X and Y coordinates
            if re.search(r'_rlnCoordinateX', line):
                x_idx = int(str(line.split()[1]).strip("#"))
                
            if re.search(r'_rlnCoordinateY', line):
                y_idx = int(str(line.split()[1]).strip("#"))
            
            # if line has mrc_base and @ sign then it's a particle.star file
            #if (re.search(rf"{mrc_base}", line) and re.search(r'@', line)):
            if re.search(rf"{mrc_base}", line):
                if x_idx + y_idx > 0:
                    x = float(line.split()[x_idx-1])
                    y = float(line.split()[y_idx-1])
                    coords.append([x,y])
                   
            # if inside loop_ section and the line does not have underscore then is a _picks.star file
            #elif (in_loop == True and not re.search(r'_', line)):
            #    if x_idx + y_idx > 0:
            #        x = float(line.split()[x_idx-1])
            #        y = float(line.split()[y_idx-1])
            #        coords.append([x,y])
               
    return coords     

if __name__ == "__main__":
   show_picks()