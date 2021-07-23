#!/bin/bash
# script to create conda environment and install necessary packages

# create conda environment
CONDAENV="apbc"
conda create --name ${CONDAENV} -y

# install conda packages
conda install -n ${CONDAENV} cudnn=7.6.5 tensorflow-gpu=2.4.1 -y
conda install -n ${CONDAENV} numpy=1.19.2 -y
conda install -n ${CONDAENV} matplotlib=3.3.4 -y
conda install -n ${CONDAENV} pillow=8.2.0 -y
conda install -n ${CONDAENV} progress=1.5.0 -y
conda install -c conda-forge -n ${CONDAENV} mrcfile=1.3.0 -y