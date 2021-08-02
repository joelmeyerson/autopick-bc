## Autopick-BC

Classification module for protein image detection in cryo-EM data. The project uses TensorFlow and can be run using the provided Docker image or it can be installed locally with conda. It is designed to be used with the Relion cryo-EM image processing program. 

### Description

Single particle cryo-EM datasets contain large format images with hundreds of individual proteins in different orientations (particles). The particles must be segmented from the images for further processing and structure determination. This is typically done using an LoG filter or using template matching with a low-pass filtered structural model. Segmentation (autopicking) typically produces many false positives and missed proteins. The false positives are removed during the next stage of processing which is 2D classification and averaging of particles. This 2D analysis step yields a collection of class averages which are manually designated as "good" or "bad". Particles composing the good classes are retained for further analysis, and particles from the bad classes are discarded.

Autopick-BC is designed as an additional step in a cryo-EM image processing pipeline. Rather than discarding the bad particles after 2D classification and averaging, it uses both the bad and good particles to train a binary classifier. After training, the classifier is used to re-pick particles from the original images to identify true and false particles. The rationale is that the approach may increase recovery of true positive particles and later improve 3D structural results.

The image classification model is a basic CNN with four convolutional layers, four max pooling layers, two fully connected layers, and one sigmoid output layer. It uses ReLU activation, the Adam optimizer, binary crossentropy loss, dropout and early stopping. A random rotation of +/-20% is used in data augmentation.

### Procedure

Autopick-BC was designed to be incorporated in any new or existing Relion project. In the Relion project directly it will make a ClassBin sub-directory to contain all associated data and metadata.

1. Create a new Relion project or use an existing project.

2. Follow the standard image processing pipeline: import movies or images, run motion correction if needed, run CTF estimation, run autopick, extract particles, run 2D classification and averaging.

3. Select good and bad classes in the Relion GUI and save them in different `particles.star` files. For example, `particles_good.star` and `particles_bad.star`. 

4. Generate training/validation and testing datasets from the good and bad particle star files. A 80/20 split is used for training/testing, and training is in turn divided with an 80/20 split to create a validation subset (64%/16% of the total particles). The outputs are `train` and `test` directories containing particle images. This step uses `gen_data.py`.

5. Train a classification model to predict whether a particle is `good` or `bad`. The output is `model.h5`. This step uses `gen_model.py`.

6. For each image in the dataset use a sliding window to tile the image into samples. The sliding window sampling interval is defined by the box size provided (typically ~1.5x the maximum particle diameter). Sampling is done each time the window slides a distance box/2 pixels which creates overlapping samples. The output is one `grid.star` for each image in the dataset. Each star files contains the same XY coordinates which are arranged in the grid pattern produced by the sliding window. This step uses `gen_grid.py`.

7. Segment each image in the dataset with coordinates specificied in the `grid.star` files. The output is one set (stack) of image segments for each input image. The paths and metadata for all segments are stored in a single `particles.star` file that indexes the image stacks. This step uses Relion to read the `grid.star` files and extract CTF corrected segments.

8. Each segment listed in the `particles.star` file is provided to the classifier model to predict whether it contains a particle or not. Segments are classified in batches. The outputs are `particles_true.star` and `particles_false.star`. This step uses the `gen_picks.py` module.

9. The results can be visualized using `show_picks.py` which generates an overlay of XY coordinates for the true or false classes on the cryo-EM image. The inputs are an image file (mrc format) and `particles_true.star` or `particles_false.star`. The output is a png file with coordinates overlayed on the image.

10. The `particles_true.star` file can be directly used for structure determination with Relion or CryoSparc.

### Running with Docker

The parent image is the `tensorflow:latest-gpu` image. The Dockerfile specifies the additional conda packages needed for Autopick-BC, and sets up a conda env.

`git clone https://github.com/joelmeyerson/autopick-bc.git`

`cd ./autopick-bc`

`docker build -t apbc .` # builds container called apbc

`cd /path/to/relion/project`

`docker run --gpus all --rm -ti -v $(pwd):$(pwd) apbc` # launch interactive container

`cd /path/to/relion/project`

`conda activate apbc` # activate the conda environment

### Running with local installation

If running locally it`s best to use GPUs and an up-to-date Nvidia driver.

`git clone https://github.com/joelmeyerson/autopick-bc.git # clone the repository

`echo 'export PATH=$PATH:/path/to/autopick-bc' >> ~/.bashrc`

`conda create --name apbc --file requirements.txt` # create conda environment, install packages

`bash -x -e build.sh` # set up conda environment

`conda activate apbc` # activate the conda environment

`cd /path/to/relion/project`


If installation fails using requirements.txt then it can be done with the included script.

`bash autopick-bc/conda/create-conda-env.sh` # create conda environment and install packages

### Training and testing with manually labeled Beta-galactosidase data

A Beta-galactosidase dataset (EMPIAR-10017) was used for development and testing. All the images in the dataset were manually labeled with positive labels (particles) and negative labels (ice chunks, carbon, empty areas). Labels are stored in the `.box` format and found in the `autopick-bc/beta-galactosidase` directory. A total of 10,497 positive labels and 4,830 negative labels were made for the 84 images in the dataset. Labels for the first image in the dataset are shown. The dataset pixel size is 1.77 Å.

![alt text](https://github.com/joelmeyerson/autopick-bc/blob/main/img/beta-gal-pos_and_neg_labels.png?raw=true)

1. Create a new Relion project. Run CTF estimation, import box files, and extract particles for "good" and "bad" coordinates.

2. Create training/validation and test datasets. This step creates the `ClassBin` directory which holds the datasets. The inputs are the Relion project path, and the two `particles.star` files.

`python /app/gen_data.py -p /path/to/relion/project -g /path/to/good/particles.star -b /path/to/bad/particles.star`

3. Train model. Creates a model (hd5) file and a table with the results of training/testing. The only input is the Relion project path.

`python /app/gen_model.py -p /path/to/relion/project`

![alt text](https://github.com/joelmeyerson/autopick-bc/blob/main/img/beta-gal-train_and_test_results.png?raw=true) 

### Results with T20 proteasome data

The T20 proteasome dataset was tested (EMPIAR-10025). Motion-corrected super-resolution images (7420x7676 dimensions, 0.6575 Å/pix) were two-fold binned (3710x3838 dimensions, 1.315 Å/pix) then imported into Relion. Autopick was run (LoG: 90 Å inner diameter; 160 Å outer diameter; 5 upper threshold), particles extracted with a box size of 216 and binned to 64 then processed with 2D classification and averaging.

Subsets of good (51,504) and bad (19,396) particles were selected in Relion and used to build datasets for training/validation.

![alt text](https://github.com/joelmeyerson/autopick-bc/blob/main/img/t20-classes.png?raw=true)

Datasets were generated and processed as follows.

`python gen_data.py -p /path/to/relion/project -g /path/to/good/particles.star -b /path/to/bad/particles.star`

`python gen_model.py -p /path/to/relion/project`

`python gen_grid.py -p /path/to/relion/project -i /path/to/micrograph.star -x boxsize`

Sliding window segmentation gave a total of 113,288 coordinates in grid.star files. These files were used in Relion to extract particles (216 extraction binned to 64).

`python gen_picks.py -p /path/to/relion/project -i /path/to/particles.star`

The results of model training/testing are shown below.

![alt text](https://github.com/joelmeyerson/autopick-bc/blob/main/img/t20-train_and_test_results.png?raw=true) 

The prediction results were evalualated by inspecting overlays of the coordinates for true and false particles on cryo-EM images. The results show that many true positives were selected, but its clear some false negatives and false positives were also selected. It is notable that empty areas frequently do are not marked as positive, which suggests the model can successfully discriminate between true particles and empty areas. No further structural analysis was done. Two examples are shown below.

![alt text](https://github.com/joelmeyerson/autopick-bc/blob/main/img/t20-pos_and_neg_labels.png?raw=true) 

### Results with urease data

A dataset with jackbean urease (EMPIAR-10656) was tested. A subset of 37 images were selected for processing. The images are super-resolution with 0.5175 Å pixel size (super-resolution) and were two-fold binned to 1.035 Å pixel size before importing into Relion. Particles were autopicked (LoG: 120 Å inner diameter; 130 Å outer diameter; 5 upper threshold), extracted with a box size of 224 downsampled to 64, and processed with 2D classification and averaging. From the results 3,543 "good" and 877 "bad" particles were selected for training and testing.

![alt text](https://github.com/joelmeyerson/autopick-bc/blob/main/img/urease-classes.png?raw=true)

Training and testing gave the following results.

![alt text](https://github.com/joelmeyerson/autopick-bc/blob/main/img/urease-train_and_test_results.png?raw=true) 

A total of 333,000 particles were predicted by the model. Inspection of the true and false particle coordinates suggested a low rate of false positives, but also showed a poor ability to detect true positives.

### Conclusions

The rationale for this approach is that it could help improve recovery of true positive particles. The results show that a CNN can be readily trained to discriminate between particle and non-particle. However, the sliding window segmentation and prediction do not appear to yield optimal results. This is evident from displaying coordinates for particles predicted to be positive or negative overlayed on the cryo-EM images. The model correctly predicts many true positives and true negatives, but there are also many false positives and negatives. A key impression is that for many true positives the segment (box) is well centered on the particle. This suggests that the model was not good at coping with off-centered particles. Data augmentation with an added offset did not seem to help this, but more testing is needed. Another impression is that using four-fold binned particles for training tends to give better training statistics, but this also needs more testing. The datasets tested here are both "ideal" in the sense that the particles have high symmetry, are large and rigid, and are generally well-selected using standard LoG and template based approaches. The particles fields in the images are also dense with particles, which limits the number of potential false positives during autopicking. For these reasons it would be valuable to test datasets that are less ideal and therefore more challenging to traditional particle picking methods.

### Limitations and update goals

Currently only ideal datasets have been tested. It will be valuable to test more challenging datasets followed by 3D structure determination.

Training/validation and testing datasets are created by converting MRC particles to PNG particles. This is inefficient because of data duplication, but has the advantage of making the data easily loadable into a TF Dataset using the image from dataset preprocessing tool. Inefficient data handling should be addressed.

Currently grid.star files are created and then used to extract particles which are in turn used for prediction. The extracted particles are read as MRC slices into memory in batches of 32 then provided to the model. A better approach would be to avoid grid.star files and Relion extraction and instead do the sliding window segmentation and batching together at run time.

Currently single models are generated during/training testing and no hyperparameters are tested. This should be addressed by adding some sort of ensemble training/testing.
