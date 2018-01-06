FROM ubuntu:16.04

# install build and run dependencies
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
      clang \
      g++ \
      git \
      make \
      man-db \
      python3 \
      python3-toml \
      && \
    rm -rf /var/lib/apt/lists/*

# create the new flit user
RUN useradd --create-home --shell /bin/bash --gid sudo flituser && \
    echo 'flituser:flitpass' | chpasswd && \
    apt-get update && \
    apt-get install -y sudo && \
    rm -rf /var/lib/apt/lists/*

# Allow the flituser to not use a password with sudo
RUN echo "flituser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/flituser

# change to the flituser and his home directory
USER flituser
WORKDIR /home/flituser

# Download and install FLiT
RUN git clone https://github.com/PRUNERS/FLiT.git && \
    cd FLiT && \
    git checkout devel && \
    make && \
    sudo make install
    # To install in the home directory:
    #make install PREFIX=/home/flituser/install
    # But making it in the PATH is a little bit of trouble

# Setup a convenient interactive development environment
RUN sudo apt-get update && \
    sudo apt-get install -y \
      bash-completion \
      emacs \
      screen \
      sqlite3 \
      tmux \
      tree \
      vim \
      && \
    sudo rm -rf /var/lib/apt/lists/*

# Add my configuration files
#RUN git clone https://bitbucket.org/mikebentley15/configurations.git && \
#    cd configurations && \
#    make -f link-config.mk backup && \
#    make -f link-config.mk all

# Create some sandbox projects
RUN mkdir sandbox && \
    cd sandbox && \
    flit init --directory empty && \
    flit init --directory litmus-tests --litmus-tests && \
    flit init --directory 01-dotproduct && \
    cd 01-dotproduct && \
    mv tests/Empty.cpp tests/DotProduct.cpp && \
    sed -i -e 's/Empty/DotProduct/g' tests/DotProduct.cpp

    # edit FLiT/sandbox/01-dotproduct/tests/DotProduct.cpp
    ## TODO: would like to be able to limit the compilers and flags used from
    ##       the flit-config.toml file
    # make runbuild -j20
    # make run
    # flit import 

CMD ["bash"]
