#!/bin/bash

# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by
#   Michael Bentley (mikebentley15@gmail.com),
#   Geof Sawaya (fredricflinstone@gmail.com),
#   and Ian Briggs (ian.briggs@utah.edu)
# under the direction of
#   Ganesh Gopalakrishnan
#   and Dong H. Ahn.
#
# LLNL-CODE-743137
#
# All rights reserved.
#
# This file is part of FLiT. For details, see
#   https://pruners.github.io/flit
# Please also read
#   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
#
# - Redistributions in binary form must reproduce the above
#   copyright notice, this list of conditions and the disclaimer
#   (as noted below) in the documentation and/or other materials
#   provided with the distribution.
#
# - Neither the name of the LLNS/LLNL nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
# SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Additional BSD Notice
#
# 1. This notice is required to be provided under our contract
#    with the U.S. Department of Energy (DOE). This work was
#    produced at Lawrence Livermore National Laboratory under
#    Contract No. DE-AC52-07NA27344 with the DOE.
#
# 2. Neither the United States Government nor Lawrence Livermore
#    National Security, LLC nor any of their employees, makes any
#    warranty, express or implied, or assumes any liability or
#    responsibility for the accuracy, completeness, or usefulness of
#    any information, apparatus, product, or process disclosed, or
#    represents that its use would not infringe privately-owned
#    rights.
#
# 3. Also, reference herein to any specific commercial products,
#    process, or services by trade name, trademark, manufacturer or
#    otherwise does not necessarily constitute or imply its
#    endorsement, recommendation, or favoring by the United States
#    Government or Lawrence Livermore National Security, LLC. The
#    views and opinions of authors expressed herein do not
#    necessarily state or reflect those of the United States
#    Government or Lawrence Livermore National Security, LLC, and
#    shall not be used for advertising or product endorsement
#    purposes.
#
# -- LICENSE END --

set -x

exists ()
{
    command -v "$1" >/dev/null 2>&1
}

python3_has ()
{
    python3 -c "import $1" >/dev/null 2>&1
}

SCRIPT_DIR="$(pwd)/$(dirname $0)"

# Check for psql install
if  ! exists createdb || ! exists psql; then
    # Install if not present
    echo "Postgres does not seem to be installed."
    echo "Attempting install now."

    # Try different package managers
    if exists apt; then
	sudo apt install postgresql postgresql-plpython3
    elif exists apt-get; then
	sudo apt install postgresql postgresql-plpython3
    elif exists pacman; then
	sudo pacman -S postgresql postgresql-lib-python3
    elif exists yum; then
	sudo yum install postgresql-server #name-for-plpython3
    elif exists brew; then
	brew install postgresql --with-python3
	brew services start postgresql
    else
	echo "Unable to find a suitable package manager."
	echo "Please install Postgres and plpython3"
	exit -1
    fi
fi


# Check for numpy install
if  ! python3_has numpy; then
    # Install if not present
    echo "Numpy does not seem to be installed for python 3."
    echo "Attempting install now."

    Try different package managers
    if exists apt; then
    	sudo apt install python3-numpy
    elif exists apt-get; then
    	sudo apt install python3-numpy
    elif exists pacman; then
    	sudo pacman -S python3-numpy
    elif exists yum; then
    	sudo yum install python3-numpy
    elif exists brew; then
    	brew install numpy -with-ptyhon3
    else
    	echo "Unable to find a suitable package manager."
    	echo "Please install numpy for python3"
    	exit -1
    fi
fi


# Check for matplotlib install
if  ! python3_has matplotlib; then
    # Install if not present
    echo "Matplotlib does not seem to be installed for python 3."
    echo "Attempting install now."

    Try different package managers
    if exists apt; then
    	sudo apt install python3-matplotlib
    elif exists apt-get; then
    	sudo apt install python3-matplotlib
    elif exists pacman; then
    	sudo pacman -S python3-matplotlib
    elif exists yum; then
    	sudo yum install python3-matplotlib
    elif exists brew; then
	brew tap homebrew/science
    	brew install homebrew/science/matplotlib -with-ptyhon3
    else
    	echo "Unable to find a suitable package manager."
    	echo "Please install Postgres and plpython3"
    	exit -1
    fi
fi

# Check if user exists
# from http://stackoverflow.com/questions/8546759/how-to-check-if-a-postgres-user-exists
if psql -t -c '\du' | cut -d \| -f 1 | grep -qw `whoami`; then
    echo "User `whoami` already exists"
else
    echo "Creating user `whoami`"
    sudo -u postgres createuser --superuser `whoami`
fi


createdb flit "The database for collecting all FLiT results"
psql flit < "$SCRIPT_DIR/tables.sql"

wait

#add our config to postgres for matplotlib
PGDIR=$(psql flit -t -c 'select getpwd()')
if [ ! -e ${PGDIR}/matplotlibrc ]; then
    sudo -u postgres cp ${SCRIPT_DIR}/matplotlibrc ${PGDIR}/matplotlibrc
else
    if ! egrep '^backend[[:space:]]*:[[:space:]]*Agg$' ${PGDIR}/matplotlibrc; then
	echo "FLiT reporting will fail without the setting 'backend : Agg' in ${PGDIR}/matplotlibrc.  Please set before using FLiT"
    fi
fi
       
#now we need to add the user and postres to the flit group

sudo addgroup flit
sudo usermod -aG flit sawaya
sudo usermod -aG flit postgres
sudo service postgresql restart
