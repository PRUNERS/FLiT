#!/bin/bash

if [[ ! -e "~/.postgresql" ]]; then
    mkdir ~/.postgresql
fi
cd ~/.postgresql
tar -xf qfp/postgresql.tar
cd ~/qfp/results
cat "*out_" >> masterRes
`which qfp` -c "select importQFPResults('$PWD/masterRes');"
exit $?
