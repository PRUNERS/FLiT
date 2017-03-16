#!/bin/bash

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
       

