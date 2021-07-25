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
from progress.bar import Bar

# local imports
#from support_modules import extract_star_meta
import extract_star_meta

def gen_grid():
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Extract segments (boxes) from images in a grid pattern using sliding window.')
    parser.add_argument('-p', '--projpath', type=str, help='path for project', required=True)
    parser.add_argument('-i', '--imagestar', type=str, help='micrograph.star for the target micrographs)', required=True)
    #parser.add_argument('-m', '--metastar', type=str, help='any particle star file containing the target box size', required=True)
    parser.add_argument('-x', '--boxsize', type=int, help='box size (must match box size used for training/testing.)', required=True)
    args = parser.parse_args()
    
    # configure working directory
    work_dir = args.projpath
    if work_dir.endswith('/'):
        work_dir = work_dir.rstrip('/')
        
    data_dir = args.projpath + '/ClassBin'
        
    # get box size
    #meta = extract_star_meta.extract(args.metastar)
    #box = int(meta[0])
    box = args.boxsize
    
    # used to later store micrograph path name from star file
    micro_dir = ""
    
    # extract name of image stack and index for image slice
    files = [] # store files in array
    with open(args.imagestar, "r") as openfile:
        for line in openfile:
            # find the header entry which contains column number that stores micrograph name
            if re.search(r'_rlnMicrographName', line):
                img_column = int(str(line.split()[1]).strip("#"))

            # use keywords to find lines with micrograph names
            if re.search(r'(mrc | mrcs | tif | tiff)', line):
                column_str = line.split()[img_column-1]
                path = work_dir + "/" + column_str
                img = str(path.split("/")[-1])
                img = img.split(".")[0]
                
                # if micro_dir variable is empty, get path to micrograph name
                if micro_dir == "":
                    micro_dir = work_dir + "/" + str(pathlib.Path(column_str).parent)
                    
                # store micrograph path and micrograph name
                files.append([path, img])
    
    if (os.path.exists(micro_dir) == False):
        print("\nMicrographs directory not found, but needed to store star file output. Exiting.")
        exit()

    if (os.path.exists(micro_dir) == True):
        print("\nMicrographs directory found. Confirm if _grid.star files can be written to " + micro_dir + " path. (y/n)")
        yn = input()
        if yn == 'n':
            print("\nExiting without writing _grid.star files.")
        elif yn == 'y':
            pass
        else:
            print("\nInvalid selection. Exiting")
            exit()

    # convert array to tensor
    files = np.array(files)
    num_file = int(len(files))

    # read mrc header
    micrograph = mrcfile.open(files[0,0], mode=u'r', permissive=False, header_only=True);
    header = micrograph.header
    dimx = header.nx
    dimy = header.ny
    dimz = header.nz

    if dimz > 1:
        print("\nSTAR file must contain flat images not movie stacks. Exiting.")
        exit()

    lines = []
    lines.append('')
    lines.append('# version 30001')
    lines.append('')
    lines.append('data_')
    lines.append('')
    lines.append('loop_')
    lines.append('_rlnCoordinateX #1')
    lines.append('_rlnCoordinateY #2')
    lines.append('_rlnClassNumber #3')
    lines.append('_rlnAnglePsi #4')
    lines.append('_rlnAutopickFigureOfMerit #5')

    # number of samples along y and x
    boxes_y = math.floor(dimy/box)
    boxes_x = math.floor(dimx/box)

    # generate a star file entry for each coordinate in the grid
    for y in range(1, boxes_y+1):
        for x in range(1, boxes_x+1):
            lines.append(str(format((box*x)-(box/2), '.1f').rjust(10)) + ' ' + str(format((box*y)-(box/2), '.1f').rjust(10)) + '            2   -999.00000   -999.00000')
            if (x < boxes_x or dimx % box > 0.5) : # decide whether or not to add the final coordinate in a super sample row
                lines.append(str(format((box*x), '.1f').rjust(10)) + ' ' + str(format((box*y), '.1f').rjust(10)) + '            2   -999.00000   -999.00000')

    # write coordinate star file for each micrograph
    for f in files:
        with open(micro_dir + '/' + f[1] + "_grid.star", "w") as starfile:
           starfile.writelines("%s\n" % l for l in lines)

    print("\nSuccessfully generated _grid.star files in " + micro_dir + " directory.")
    
if __name__ == "__main__":
    gen_grid()