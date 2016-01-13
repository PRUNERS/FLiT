#!/bin/bash

RDIR=$($(which pwd))

echo running from $RDIR

echo 'running local test . . .'

./runTest ./test3.O2

echo 'running mic (phi) test . . .'

ssh mic0 $RDIR/runTest $RDIR/test3.O2.mic

if [[ $? != 0 ]]; then
    echo 'You need to be on a stampede work node.  Try accessing interactive node with "idev" command . . .'
fi

