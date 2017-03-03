#!/bin/bash

exists ()
{
    command -v "$1" >/dev/null 2>&1
}

SCRIPT_DIR="$(dirname $0)"


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
