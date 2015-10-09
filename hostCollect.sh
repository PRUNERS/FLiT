#!/bin/bash

#this is for Cloudlab
if [[ ! -e "~/.postgresql" ]]; then
    mkdir ~/.postgresql
    cp /local/.postgresql/* ~/.postresql/
fi

rm results/*
cd perpVects
make -j $1 -f Makefile2

cd ../results
cat "*out_" >> masterRes
$(which psql) -d qfp -h bihexal.cs.utah.edu -U sawaya -c "select importQFPResults('$PWD/masterRes');"
exit $?
