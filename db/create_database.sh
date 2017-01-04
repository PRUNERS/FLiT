#!/bin/bash

SCRIPT_DIR="$(dirname $0)"

createdb flit "The database for collecting all FLiT results"
psql flit < "$SCRIPT_DIR/tables.sql"
