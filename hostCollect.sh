#!/bin/bash

#this is for Cloudlab
if [[ ! -e "~/.postgresql" ]]; then
    mkdir ~/.postgresql
    cp /local/.postgresql/* ~/.postresql/
fi

rm results/*
git stash
git checkout master
git pull
cd perpVects
make -j $1 -f Makefile2

cd ../results
cat "*out_" >> masterRes
$(which qfp) -c "select importQFPResults('$PWD/masterRes');"
exit $?
