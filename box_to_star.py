#!/usr/bin/env python3

import os
import shutil
import argparse
import sys
import re
import pathlib
import math

def main():
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Convert box file to star file.')
    parser.add_argument('-f', '--boxfile', type=str, help='box file name', required=True)
    parser.add_argument('-o', '--starfile', type=str, help='star file name end in _path.star', required=True)
    args = parser.parse_args()
    
    # check that input file exists
    if os.path.isfile(args.boxfile) == False:
        print("Input box file does not exist. Exiting.")
        exit()
        
    # check that path for output file exists
    outpath = pathlib.Path(args.starfile).parent
    if (os.path.exists(outpath) == False):
        print("Output file path does not exist. Exiting.")
        exit()
    
    # get lines in box and store in list
    boxes = []
    with open(args.boxfile, "r") as openfile:
        for line in openfile:
            
            if bool(re.search(r'\d', str(line))):
                boxes.append(line.split())

    # get boxsize from first line in box file
    boxsize = int(boxes[0][3])
    
    with open(args.starfile, "w") as starfile:
        
        starfile.writelines('\n')
        starfile.writelines('# version 30001\n')
        starfile.writelines('\n')
        starfile.writelines('data_\n')
        starfile.writelines('\n')
        starfile.writelines('loop_\n')
        starfile.writelines('_rlnCoordinateX #1\n')
        starfile.writelines('_rlnCoordinateY #2\n')
        starfile.writelines('_rlnClassNumber #3\n')
        starfile.writelines('_rlnAnglePsi #4\n')
        starfile.writelines('_rlnAutopickFigureOfMerit #5\n')
        
        for box in boxes:
            # box coord system is in bottom left of box, move to center for star format
            box_x = int(box[0]) + boxsize/2
            box_y = int(box[1]) + boxsize/2 
            
            starfile.writelines("%s" % str(format(box_x, '.1f').rjust(10)) + ' ' + str(format(box_y, '.1f').rjust(10)) + '            2   -999.00000   -999.00000\n')
    
    print("\nSuccessfully generated star file.")
    
if __name__ == "__main__":
    main()