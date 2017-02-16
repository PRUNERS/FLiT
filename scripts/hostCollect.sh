#!/bin/bash -x

#the following vars should be selectively defined prior to execution
#CORES DO_PIN(or not) DB_USER DB_HOST

set -e

mkdir -p results

#do the full test suite
cd src

make -j ${CORES}

cd ..

#do PIN
if [ ! -z ${DO_PIN} ]; then
    
    #setup PIN tool
    if [ -e pin ]; then
	rm -fr pin
    fi
    mkdir pin
    cd pin
    wget http://software.intel.com/sites/landingpage/pintool/downloads/pin-3.0-76991-gcc-linux.tar.gz
    tar xf pin*
    rm *.gz
    mv pin* pin

    pushd .
    cd pin/source/tools/SimpleExamples
    make obj-intel64/opcodemix.so
    popd

    export PINPATH=$(pwd)/pin

    #run pin tests
    cd ../results
    make -j ${CORES} -f ../scripts/MakeCollectPin
    cd ..
fi

cd results

#zip up all outputs

ZIPFILE=$(hostname)_$(date +%m%d%y%H%M%S)_flit.tgz
tar zcf ${ZIPFILE} *
scp ${ZIPFILE} ${DB_USER}@${DB_HOST}:~/flit_data

exit $?