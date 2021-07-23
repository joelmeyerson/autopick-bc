# dockerfile to install miniconda and conda dependencies in tf parent image
# build with: "docker build -t apbc ."
# specify parent image
FROM tensorflow/tensorflow:latest-gpu
SHELL ["/bin/bash", "-c"]
WORKDIR /app
#COPY ./container-entrypoint.sh /app
#COPY ./script.py /app
ENV PATH="/opt/miniconda/bin:${PATH}"
ENV CONDAENV="apbc"

# install curl
RUN apt-get update && apt-get install curl -y \

# download and install miniconda
&& curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > /tmp/Miniconda.sh \
&& chmod +x /tmp/Miniconda.sh \
&& mkdir /root/.conda \
&& /tmp/Miniconda.sh -b -p /opt/miniconda \

# initializes conda for bash shell interaction
&& conda init bash \

# upgrade conda to the latest version
&& conda update -n base -c defaults conda -y \

# create the work environment and setup its activation on start
&& conda create --name ${CONDAENV} -y \ 
&& conda install -n ${CONDAENV} cudnn=7.6.5 tensorflow-gpu=2.4.1 -y \
&& conda install -n ${CONDAENV} numpy=1.19.2 -y \
&& conda install -n ${CONDAENV} matplotlib=3.3.4 -y \
&& conda install -n ${CONDAENV} pillow=8.2.0 -y \
&& conda install -n ${CONDAENV} progress=1.5.0 -y \
&& conda install -c conda-forge -n ${CONDAENV} mrcfile=1.3.0 -y

# set up environment so container can be run non-interactively
#ENTRYPOINT ["/bin/bash","/app/container-entrypoint.sh"]