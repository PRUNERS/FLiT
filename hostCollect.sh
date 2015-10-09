#!/bin/bash

#this is for Cloudlab
if [[ ! -e "~/.postgresql" ]]; then
    mkdir ~/.postgresql
    cp /local/.postgresql/* ~/.postresql/
fi

if [[ ! -e results ]]; then
    mkdir results
fi
rm results/*
cd perpVects
make -j $1 -f Makefile2

cd ../results
tar -zxf *.tgz
cat "*out_" >> masterRes_$(HOST)
git add masterRes_$(HOST)
git commit -m 'data collecton from $(HOST)'
git push
# $(which psql) -d qfp -h bihexal.cs.utah.edu -U sawaya -c "select importQFPResults('$PWD/masterRes');"
exit $?
