#!/bin/bash -x

#the following vars should be selectively defined prior to execution
#CORES DO_PIN(or not) DB_USER DB_HOST DB_PASSWD

#set -e

echo cores: ${CORES}
echo DO_PIN: ${DO_PIN}
echo DB_USER: ${DB_USER}
echo DB_HOST: ${DB_HOST}
echo FLIT_DIR: ${FLIT_DIR}
echo SLURMED: ${SLURMED}
echo CUDA_ONLY: ${CUDA_ONLY}

mkdir -p results

#do the full test suite
cd src

if [ "$CUDA_ONLY" = "False" ]; then
    unset CUDA_ONLY
fi

make -j ${CORES} > ../results/makeOut

cd ..

#do PIN
if [ "${DO_PIN}" = "True" ]; then
    
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
    make -j ${CORES} -f ../scripts/MakeCollectPin >> makeOut
    cd ..
fi

cd results

#zip up all outputs
ZIPFILE=$(hostname)_$(date +%m%d%y%H%M%S)_flit.tgz
tar zcf ${ZIPFILE} *

if [ "${SLURMED}" != "None" ];
then
    scp ${ZIPFILE} ${DB_USER}@${DB_HOST}:~/flit_data
fi

exit $?
