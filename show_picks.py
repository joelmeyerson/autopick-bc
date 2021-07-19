import sys
import re
import numpy as np
import mrcfile
from PIL import Image, ImageEnhance, ImageDraw
import argparse

CONTRAST_FACTOR = 4
CIRCLE_LINE_WIDTH = 10
BIN_FACTOR = 2

def gen_picks_display():

    # parse arguments
    parser = argparse.ArgumentParser(description='Generate overlay of coordinates on image.')
    parser.add_argument('-p', '--projpath', type=str, help='path for project', required=True)
    parser.add_argument('-i', '--image', type=str, help='image file in mrc format', required=True)
    parser.add_argument('-c', '--starcoord', type=str, help='input star file with good particles', required=False)
    parser.add_argument('-x', '--boxsize', type=str, help='box size to display', required=False)
    
    args = parser.parse_args()
    
    mrc = mrcfile.open(args.image, mode=u'r', permissive=False, header_only=False)

    # check that mrc files are flat (not stacks)
    if mrc.data.ndim > 2:
        print("Images must be flat (Z = 1). Cannot process multi-layer images.")
        exit()
    
    # process image
    img_array = np.flip(mrc.data, axis=0)
    img_array = img_array + abs(img_array.min()) # make all 32 bit floating point pixel values >= 0
    img_array /= img_array.max() # normalize all pixels between 0 and 1
    img_array *= 255 # normalize all pixels between 0 and 255

    img_base_name = args.image.split('/')[-1].split('.')[0] # get file base name
    image = Image.fromarray(img_array).convert("L")
    image = ImageEnhance.Contrast(image).enhance(CONTRAST_FACTOR)
    
    # add coordinates overaly to image
    draw = ImageDraw.Draw(image)
    x_column_idx = 0
    y_column_idx = 0
    x_val = 0.0
    y_val = 0.0
    loop_flag = False # used to check when pass loop flag in star file
    
    box = int(args.boxsize)
    with open(args.starcoord, "r") as openfile:
        for line in openfile:
            # find the header entry which contain X and Y coordinates
            if re.search(r'_rlnCoordinateX', line):
                x_column_idx = int(str(line.split()[1]).strip("#"))
                
            if re.search(r'_rlnCoordinateY', line):
                y_column_idx = int(str(line.split()[1]).strip("#"))
                
            if line.lstrip().startswith("loop"):
                loop_flag = True
            
            elif loop_flag == True and not line.startswith("_"): # now in data table portion
                x_val = float(line.split()[x_column_idx-1])
                y_val = float(line.split()[y_column_idx-1])
                draw.ellipse([x_val-box/4, y_val-box/4, x_val+box/4, y_val+box/4], fill=None, outline='black', width=CIRCLE_LINE_WIDTH) # make circle 1/2 box size
    # write png
    image.save(args.projpath + '/ClassBin/' + img_base_name + '_pick.png')
    mrc.close()

if __name__ == "__main__":
   gen_picks_display()