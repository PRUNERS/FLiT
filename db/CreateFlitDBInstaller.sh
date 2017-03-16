#!/bin/bash 

#this creates a script for execution on the desired DB Host.
#The script will contain files needed for its configuration,
#currently, setup_db_host.sh and matplotlibrc

DIR=$(dirname "$(readlink -f "$0")")
FILE=InstallFlitDB.sh
INFILE=${DIR}/${FILE}.in
PREFIX=$RANDOM

echo "#!/bin/bash" > $FILE

echo -n "TAGS=(" >> $FILE

for ifile in "$@"
do
    echo -n $ifile ' ' >> $FILE
done

echo ")" >> $FILE

echo "PREFIX="$PREFIX >> $FILE

cat $INFILE  >> $FILE

for ifile in "$@"
do
    echo ${PREFIX}$ifile >> $FILE
    cat $ifile >> $FILE
done
