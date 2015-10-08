#!/bin/bash

if [[ ! -e "~/.postgresql" ]]; then
    mkdir ~/.postgresql
    cp /local/.postgresql/* ~/.postresql/
fi
cd ~/qfp/results
cat "*out_" >> masterRes
$(which qfp) -c "select importQFPResults('$PWD/masterRes');"
exit $?
