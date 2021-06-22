# Autopick-BC

Foobar is a Python library for dealing with word pluralization.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)






docker-container-build
-- Docker file
-- container-entrypoint.sh
-- copy script files to /app


environment.yml
+ make with conda env export --from-history --name autopick-bc > environment.yml
+ generate condo environment using # run: conda env create --file environment.yml

beta-galactosidase
-- box files
-- model

main.py
extract_particles.py
extrac_meta_data.py


README.md
-- program description
----- Classifies results from autopick into two classes. The program augments 2D or 3D classification by providing a second pass autopick route. Essentially, asks is this a particle or not.
----- Designed to work with the STAR file format. Was conceived to be a module in the Relion project workflow and can be introduced to any existing Relion project. The program is unobtrusive and creates only one directory in the Relion folder (ClassBin). The output is a set of coordinates that can be used for particle extraction and further processing.

-- Sample dataset
--- The program was developed using the Beta galactosidase dataset from the Scheres lab which contains X motion-corrected images (link). The data were labeled by designating true particles in the images (positive labels), and by designating empty ice, crystalline ice, ethane contamination, and carbon support film (negative labels).

-- Running
----- 1. Set up Conda
-- Running with Docker container (interactive mode)

-- Running with Docker container (executable mode)
---- to be implemented



###########################

workflow:

1. Train/test using data from 2DC. Output is a of training/testing results, table of accuracy, and a model.

2. Segment micrographs to extract data to predict with model. Input should be coordinate file (grid),  micrograph star file. Output should be images which can be used for prediction.

3. Run prediction on images. Output should be a STAR coordinate file with "true" particles.


PROJECT
- ClassBin
-- train_good
-- train_bad
-- test_good
-- test_bad
-- grid.star
-- model.hd5








Try training without CTF estimation?

Fix accuracy metricsUse pandas to plot out put and write to image fileImplement sliding window

Try using crop during training

conda install numpy=1.19.2

Conda requirements

docker run --gpus all --rm -ti -v $(pwd):/home/joel/pick-test autopick-bc-container

conda env export --from-history --name autopick-bc > autopick-bc.yml