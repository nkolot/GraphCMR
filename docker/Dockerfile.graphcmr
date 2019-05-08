ARG BASE_NUMBER
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG PYTHON_TAG

FROM chaneyk/daniilidis-group-base:cuda${CUDA_VERSION}-${PYTHON_TAG}-${BASE_NUMBER}

ARG PYTHON_VERSION
ARG PYTHON_FULL_VERSION
ARG TORCH_VERSION

LABEL maintainer="Kenneth Chaney <chaneyk@seas.upenn.edu>"

ENV CUDADIR /usr/local/cuda
ENV OPENBLASDIR /usr/lib/x86_64-linux-gnu/openblas
ENV GPU_TARGET = Kepler Maxwell Pascal Volta
WORKDIR /opt

RUN apt-get update && apt install -y gfortran  libopenblas-dev && \
    wget http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.4.0.tar.gz && \
    tar xzf magma-2.4.0.tar.gz
WORKDIR /opt/magma-2.4.0
RUN cp make.inc-examples/make.inc.openblas make.inc && make -j 8

ENV LD_LIBRARY_PATH /usr/local/magma/lib:$LD_LIBRARY_PATH

WORKDIR /opt
RUN git clone https://github.com/pytorch/pytorch
WORKDIR /opt/pytorch

RUN git checkout v${TORCH_VERSION}
RUN git submodule update --init
RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="/opt/magma-2.4.0" \
    pip${PYTHON_VERSION} install -v .

WORKDIR /opt
RUN git clone https://github.com/pytorch/vision.git && cd vision && git checkout v0.2.2 && pip${PYTHON_VERSION} install -v .

RUN pip${PYTHON_VERSION} --no-cache-dir install chumpy tensorboardX neural_renderer_pytorch

RUN if [ -z "${PYTHON_VERSION}" ]; then  pip${PYTHON_VERSION} --no-cache-dir install opendr==0.77; fi

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

WORKDIR /opt
RUN wget https://spdf.gsfc.nasa.gov/pub/software/cdf/dist/cdf37_0/cdf-dist-all.tar.gz && gunzip cdf-dist-all.tar.gz && tar -xvf cdf-dist-all.tar
WORKDIR /cdf37_0-dist
RUN make OS=linux ENV=gnu CURSES=no all &&  make INSTALLDIR=/usr/local/cdf install && make clean

WORKDIR /

RUN rm -rf /usr/local/lib/python2.7/dist-packages/torch-* && rm -rf /usr/local/lib/python3.6/dist-packages/torch-*

RUN pip${PYTHON_VERSION} --no-cache-dir install spacepy
