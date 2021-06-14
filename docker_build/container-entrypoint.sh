#!/bin/bash
# enter conda environment
conda init bash
source activate bc-autopick

# run program
# use the next line to run the classifier; requires that main.py sits in /app of the container; pass command line arguments entered in the docker call by using bash $1, $2, etc. or use Python args.
python /app/main.py
