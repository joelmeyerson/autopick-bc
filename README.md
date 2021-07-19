# Autopick-BC

Classification module for protein image detection in cryo-EM data implemented using Keras/Tensorflow.

## Modules

gen_data.py - Generate training/validation and testing datasets from 'good' and 'bad' particle files. Main outputs are 'train' and 'test' directories containing particles images in png format.

gen_model.py - Train and test a classification model to predict whether a particle is 'good' or 'bad'. Main output is 'model.h5'.

gen_grid_star.py - Produce coordinates arranged in a grid over each micrograph (tiled). Main output is one coordinate star file per micrograph. The files are to be used for extraction of CTF corrected particles in Relion.

gen_picks.py - Use model to predict whether an image tile contains a particle or not. The input is a star file containing particles extracted from the grid coordinates. Particles are read in MRC and processed in batches which avoids having to convert to PNG. Output is set of coordinates for particles and non-particles.

gen_show_picks.py - Visualize micrographs with overlays of particles and non-particles.
