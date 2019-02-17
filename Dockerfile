FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         locales \
         cmake \
         git \
         curl \
         vim \
         unzip \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libfreetype6-dev \
         libxft-dev &&\
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=2.7 numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN conda install -c pytorch pytorch=0.4.1 cuda90

RUN conda install flask=0.12.2=py27_0 && \
    conda install flask-socketio=2.8.5=py27_0 && \
    conda install nltk=3.2.4=py27_0 && \
    conda install numpy=1.13.3=py27hdbf6ddf_4 && \
    conda install pandas=0.20.3=py27_0 && \
    conda install ujson=1.35=py27_0 && \
    conda install decorator=4.1.2=py27_0 && \
    conda install matplotlib=2.0.2=np113py27_0

RUN pip install future==0.16.0 && \
    pip install nose==1.3.7 && \
    pip install scikit-learn==0.19.0 && \
    pip install sklearn==0.0 && \
    pip install torchtext==0.2.1 && \
    pip install visdom==0.1.6.1

RUN python -m nltk.downloader punkt && \
    python -m nltk.downloader stopwords

RUN DUMMY3=${DUMMY3} git clone https://github.com/stanfordnlp/cocoa.git && \
    cd cocoa && \
    python setup.py develop
