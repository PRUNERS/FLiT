#!/bin/bash -x

mkdir -p results

#do the full test suite
cd qfpc
if [[ $2 == True ]]
then
    export CUDA_ONLY=True
fi
make -j $1

#setup pin tool
cd ..
if [[ -e pin_tool ]]; then
    rm -fr pin_tool
fi
mkdir pin_tool
cd pin_tool
wget http://software.intel.com/sites/landingpage/pintool/downloads/pin-3.0-76991-gcc-linux.tar.gz
tar xf pin*
rm *.gz
mv pin* pin
export PINPATH=$(pwd)/pin

#run pin tests
cd ../results
make -j $1 -f ../pin/MakeCollect

#zip up all outputs
tar zcf $(hostname)_flit.tgz *

exit $?
