#!/usr/bin/env python3
'Generates statistics from a given test results csv file'

import csv
import sys
import argparse

def main(arguments):
    'Main entry point'
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='''
            Generates statistics from a given test results csv file.  The csv
            file should have columns "switches", "precision", "sort", "score0",
            "score0d", "host", "compiler", "name", "score1", "score1d"
            ''',
        )
    parser.add_argument('csvfile')
    args = parser.parse_args(args=arguments)

    # Read the csvfile into memory
    rows = []
    with open(args.csvfile, 'r') as infile:
        reader = csv.DictReader(infile)
        rows = [x for x in reader]

    # Generate statistics
    names = set(x['name'] for x in rows)
    groups = {}
    scores = {}
    for name in names:
        groups[name] = [x for x in rows if x['name'] == name]
        scores[name] = set(x['score0'] for x in groups[name])
        print(name, len(scores[name]))
    print()

    

    total_switches = set(x['switches'] for x in rows)
    for switch in total_switches:
        switch_counts[switch] = 0
    for name in names:
        switches = set(x['switches'] for x in groups[name])
        ## TODO: Find a base score to compare against...
        ## TODO:   Without that, we don't have anything to plot on the switches front
        ## TODO:   Maybe we could have the base be
        ## TODO:   - the most common score.
        ## TODO:   - the empty switch for gcc
        ## TODO:   - something else
        #for switch in switches:
        #    switch_counts[switch] += len(set(x['score0'] for x in 

if __name__ == '__main__':
    main(sys.argv[1:])

