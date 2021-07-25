## Autopick-BC

Classification module for protein image detection in cryo-EM data. The project uses TensorFlow and can be run using the provided Docker container or it can be installed locally with conda. It is designed to be used with the Relion cryo-EM image processing program. 

### Description

Single particle cryo-EM datasets contain large format images with hundreds of individual proteins in different orientations (particles). The particles must be segmented from the images for further processing and structure determination. This is typically done using an LoG filter or using template matching with a low-pass filtered structural model. Segmentation (autopicking) typically produces many false positives and missed proteins. The false positives are removed during the next stage of processing which is 2D classification and averaging of particles. This 2D analysis step yields a collection of class averages which are manually designated as "good" or "bad". Particles composing the good classes are retained for further analysis, and particles from the bad classes are discarded.

Autopick-BC is designed as an additional step in a cryo-EM image processing pipeline. Rather than discarding the bad particles, it uses both the bad and good particles from 2D class averaging to train a binary classifier. After training, the classifier is used to re-pick particles from the original images to identify true and false particles. The rationale is that the approach will increase recovery of true positive particles and later improve 3D structural results.

### Procedure

Autopick-BC was designed to be incorporated in any new or existing Relion project. In the Relion project directly it will make a ClassBin sub-directory to contain all associated data and metadata.

1. Create a new Relion project or use an existing project.

2. Follow the standard image processing pipeline: import movies or images, run motion correction if needed, run CTF estimation, run autopick, extract particles, run 2D classification and averaging.

3. Select good and bad classes in the Relion GUI and save them in different `particles.star` files. For example, `particles_good.star` and `particles_bad.star`. 

4. Generate training/validation and testing datasets from the good and bad particle star files. The outputs are `train` and `test` directories containing particle images. This step uses `gen_data.py`.

5. Train and test a classification model to predict whether a particle is `good` or `bad`. The output is `model.h5`. This step uses `gen_model.py`.

6. For each image in the dataset use a sliding window to tile the image into samples. The output is one `grid.star` for each image in the dataset. Each star files contains the same XY coordinates which are arranged in grid pattern produced by the sliding window. This step uses `gen_grid.py`.

7. Segment each image in the dataset with coordinates specificied in the `grid.star` files. The output is one set (stack) of image segments for each input image and a single `particles.star` file that indexes the image stacks. This step uses Relion to read the `grid.star` files and extract CTF corrected segments.

8. Each segment listed in the `particles.star` file is provided to the classifier model predict whether it contains a particle or not. Segments are classified in batches. The outputs are `particles_true.star` and `particles_false.star`. This step uses the `gen_picks.py` module.

9. The results can be visualized using `show_picks.py` which generates an overlay of XY coordinates for the true or false classes on the cryo-EM image.

10. The `particles_true.star` file can be directly used for structure determination with Relion or CryoSparc.

### Running with Docker

The parent image is the `tensorflow:latest-gpu` image. The Dockerfile specifies the additional Conda packages needed for Autopick-BC, and sets up a Conda env.

`git clone https://github.com/joelmeyerson/autopick-bc.git` # clone the repository

`cd ./autopick-bc`

`docker build -t apbc .` # builds container called apbc

`cd < relion-project >` # where < relion-project > is the path to the Relion project directory

`docker run --gpus all --rm -ti -v $(pwd):$(pwd) apbc` # launch interactive container

`cd <relion-project>`

`conda activate apbc` # activate the Autopick-BC conda environment

### Running with local installation

If running locally it`s best to use GPUs and an up-to-date Nvidia driver.

`git clone https://github.com/joelmeyerson/autopick-bc.git # clone the repository

`echo 'export PATH=$PATH:< autopick-bc >' >> ~/.bashrc` # where < autopick-bc > is the path to the Autopick-BC directory

`conda create --name apbc --file requirements.txt` # create conda environment, install packages

`bash -x -e build.sh` # set up conda environment

`conda activate apbc` # activate the Autopick-BC conda environment

`cd < relion-project >` # where < relion-project > is the path to the Relion project directory


If installation fails using requirements.txt then it can be done with the included script.

`bash autopick-bc/conda/create-conda-env.sh` # create conda environment and install packages

### Training and testing with manually labeled Beta-galactosidase data

The Scheres lab Beta-galactosidase dataset (EMPIAR-10017) was used for development and testing. All the images in the dataset were manually labeled with positive labels (particles) and negative labels (ice chunks, carbon, empty areas). Labels are stored in the `.box` format and found in the `autopick-bc/beta-galactosidase` directory.

1. Created a new Relion project. Ran CTF estimation, imported box files, and extracted particles for "good" and "bad" coordinates. Examples of good and bad labels below.

2. Create training/validation and test datasets. This step creates the `ClassBin` directory which holds the datasets. The inputs are the Relion project path, and the two particles.star files.

`python gen_data.py -p /path/to/relion/project -g /path/to/good/particles.star -b /path/to/bad/particles.star`

3. Train model. Creates a model (hd5) file and a file with the results of training/testing. The only input is the Relion project path.

`python gen_model.py -p /path/to/relion/project`

![alt text](https://github.com/joelmeyerson/autopick-bc/blob/main/img/train_and_test_results.png?raw=true)

### Results with Beta-galactosidase data

The Beta-galactosidase dataset was also tested without manually labeled data. In this test, images were subjected to LoG autopick, 2D classification and averaging, and "good" and "bad" particle subsets were used to train a new binary classifier model. After classification, images were segmented using a sliding window (grid), and then the classifier model was used to look for true and false particles. The true particles were then used for structure determination.

In Relion run autopick and generate 2D class averages. Settings were taken from the Relion tutorial on Beta-galactosidase (LoG autopick: 150 Å inner diameter; 180 Å outer diameter; 5 upper threshold. Particle extraction: 256 box size; 64 binned box size. 2DC: 50 classes; 350 Å mask diameter; no fast subsets). Select "good" and "bad" particles.

`python gen_data.py -p /path/to/relion/project -g /path/to/good/particles.star -b /path/to/bad/particles.star`

`python gen_model.py -p /path/to/relion/project`

`python gen_grid.py -p /path/to/relion/project -i /path/to/micrograph.star -x boxsize`

In Relion use the grid.star coordinate file to extract particles.

`python gen_picks.py -p /path/to/relion/project -i /path/to/particles.star`

Results of structure determination.

TBD

### Results with T20 proteasome data

TBD

### Conclusions

TBD

### Update goals

Currently grid.star files are created and then used to extract particles which are in turn used for prediction. The extracted particles are read as MRC slices into memory in batches of 32 then provided to the model. A better approach would be to avoid grid.star files and Relion extraction and instead do the sliding window segmentation and batching together 

Training/validation and testing datasets are created by converting MRC particles to PNG particles. This is inefficient because of data duplication, but has the advantage of making the data easily loadable into a TF Dataset using the image from dataset preprocessing tool. Inefficient data handling should be addressed.

Data binning needs to be tested and implemented.
