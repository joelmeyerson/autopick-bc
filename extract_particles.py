import sys
import re
import numpy as np
import mrcfile
from PIL import Image
from progress.bar import Bar

def extract(projpath, starfile, train_dir, test_dir, good_or_bad):
    
    # extract name of image stack and index for image slice
    files = []
    with open(starfile, "r") as openfile:
        for line in openfile:
            # find the header entry which contains column number that stores image name
            if re.search(r'_rlnImageName', line):
                img_column = int(str(line.split()[1]).strip("#"))

            # use keywords to find lines with image names
            if re.search(r'(mrc | mrcs)', line):
                column_str = line.split()[img_column-1]
                path = projpath + "/" + str((column_str.split("@", 1)[1]))
                img = str(path.split("/")[-1])
                idx = int((column_str.split("@", 1)[0].lstrip("0")))

                #store path, stack name, and image slice index
                files.append([path, img, idx])

    # store image slices in a tensor then write images to disk
    files = np.array(files)
    num_file = int(len(files))

    # set fractions to use for training and testing (validation data fraction set within TF)
    num_file = int(len(files))
    train_fraction = round(num_file * 0.8)
    
    # create progress bar
    bar = Bar('Extracting ' + str(num_file) + ' ' + good_or_bad + ' particles:', fill='#', suffix='%(percent)d%%', max=num_file)
    
    # extract each particle from its MRC stack and convert to PNG
    for i in range(num_file):
        mrc_path = files[i, 0]
        mrc_file = files[i, 1]
        mrc_slice = int(files[i, 2]) - 1
       
        mrc = mrcfile.open(mrc_path, mode=u'r', permissive=False, header_only=False)

        # handle case where mrc stack has only one image in the stack so has only two dimensions
        if mrc.data.ndim == 2:
            img_array = np.flip(mrc.data[:, :])
        else: # it has three dimensions
            img_array = np.flip(mrc.data[mrc_slice, :, :])
            
        img_array = np.flip(mrc.data[mrc_slice, :, :], axis=0)
        img_array = img_array + abs(img_array.min()) # make all 32 bit floating point pixel values >= 0
        img_array /= img_array.max() # normalize all pixels between 0 and 1
        img_array *= 255 # normalize all pixels between 0 and 255
        
        # write image
        mrc_base_name = mrc_file.split('.', 1)[0] # get file base name
        if i <= train_fraction:
            Image.fromarray(img_array).convert("L").save(train_dir + '/' + mrc_base_name + '-' + str(mrc_slice) + '.png')
        else:
            Image.fromarray(img_array).convert("L").save(test_dir + '/' + mrc_base_name + '-' + str(mrc_slice) + '.png')
        
        mrc.close()
        bar.next()
    bar.finish()
