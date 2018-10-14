FROM heroku/heroku:16

# Python and Caffe native dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

# Building Caffe
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

ENV CLONE_TAG=1.0

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git .
RUN pip install --upgrade pip
RUN pip install -r ./python/requirements.txt
RUN mkdir build && cd build && \
    cmake -DCPU_ONLY=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig


# Install NSFW model and python helper
RUN cd /opt && git clone https://github.com/yahoo/open_nsfw.git
ENV PYTHONPATH /opt/open_nsfw:$PYTHONPATH
ENV PATH /opt/open_nsfw:$PATH


# Build and install app dependencies
ADD ./requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -q -r /tmp/requirements.txt
RUN pip install --ignore-installed pyOpenSSL --upgrade

# Add our code
ADD ./web /opt/web/
WORKDIR /opt/web

# Run the app.  CMD is required to run on Heroku
# $PORT is set by Heroku
CMD gunicorn --timeout 360 --bind 0.0.0.0:$PORT -k gevent --worker-connections 32 app:app
