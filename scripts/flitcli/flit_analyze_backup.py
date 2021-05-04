# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2020, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by
#   John Jacobson (john.jacobson@utah.edu)
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

'Implements the analyze subcommand'

import argparse
import os
import re
import subprocess as subp
import sys
import glob
import fileinput
import json
import math
import flitconfig as conf
import flitutil as util
import pygraphviz as pgv

brief_description = 'Aggregates and analyzes log files.'

def populate_parser(parser=None):
    'Populate or create an ArgumentParser'
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.description = '''
            Aggregates log files generated from --logging flag into one
            log file. Has utilities for getting log data in sqlite3 form
            and for plotting timing of log events.
            '''
    parser.add_argument('-C', '--directory', default='.',
                        help='The directory to initialize')

    parser.add_argument('-l', '--logs', default=False, action='store_true',
                        help='Enable logging of FLiT events.')
    parser.add_argument('-r', '--runtype', default='flit',
                        help='''
                        Type of run to analyze. Options are "flit" or
                        "bisect".
                        ''')

    return parser


def log_to_dict(log_dir):
    '''
        Read raw log files and aggregate to single log file.
        Returns a list of individual event dictionaries.
    '''

    with util.pushd(log_dir):
        #------------------------
        # Aggregate FLiT logs
        #------------------------
        # Read all log files into one final.log
        # May make more sense to process files in chunks.
        fin_list = glob.glob('flit-*.log')
        lines = []
        for fin in fin_list:
            f = open(fin, 'r')
            lines = lines + f.read().strip().split('\n')
            f.close()
#        for line in lines:
#            print(line)
#            json.loads(line)
        lines.sort(key=lambda k: json.loads(k)['time'])
        with open('final_flit.log', 'w') as fout:
            fout.write('\n'.join(lines))
       
        # Now process data read from all logs while it is in memory.
        flit_events = [json.loads(line) for line in lines]

        #------------------------
        # Aggregate bisect logs
        #------------------------
        # Read all log files into one final.log
        # May make more sense to process files in chunks.
        fin_list = glob.glob('bisect_*.log')
        lines = []
        for fin in fin_list:
            f = open(fin, 'r')
            lines = lines + f.read().strip().split('\n')
            f.close()
#        for line in lines:
#            print(line)
#            json.loads(line)
        lines.sort(key=lambda k: json.loads(k)['time'])
        with open('final_bisect.log', 'w') as fout:
            fout.write('\n'.join(lines))
       
        # Now process data read from all logs while it is in memory.
        bisect_events = [json.loads(line) for line in lines]

    return flit_events, bisect_events


def get_event_duration(events):
    '''
        Aggregates total time accumulated for each event name.
        Returns a dictionary of event names with aggregated data.
    '''
    event_times = {}

    for event in events:
        name = event['name']
        if name not in event_times:
            event_times[name] = {'start_total': 0, 'stop_total': 0,
                                 'start_count': 0, 'stop_count': 0}
        if event['start_stop'] == 'start':
            event_times[name]['start_total'] += event['time']
            # for error checking
            event_times[name]['start_count'] += 1
        else:
            event_times[name]['stop_total'] += event['time']
            # for error checking
            event_times[name]['stop_count'] += 1
   
    # Should have equal starts and stops for each event
    for key, val in event_times.items():
        assert val['start_count'] == val['stop_count'], \
               'Unequal start/stop count for event: ' + key + \
               ' starts: ' + str(val['start_count']) + ' stops: ' + str(val['stop_count'])
        assert val['start_total'] < val['stop_total'], \
               'Start time should be less than stop time for event: ' + key
 
    return event_times


def pair_events(grouped_events):
    '''
        Takes in a dict of grouped events
        Returns dict after combining start/stop pairs
    '''
    paired_events = dict()

    # go through each event type (name)
    for event_name, v in grouped_events.items():
        events = list(v)

        # for each individual event of this type, find its start/stop pairing
        while len(events) > 0:
            this_event = events.pop()
            event_match = None
            # go through events of this name to find matching message (i.e. find pair)
            for event in events:
                if event['message'] == this_event['message']:
                    event_match = event
                    time = abs(event_match['time'] - this_event['time'])
                    # keep track of time for each event
                    if event_name.strip() not in paired_events:
                        paired_events[event_name.strip()] = list()
                    paired_events[event_name.strip()].append({'message':this_event['message'], 'time':time})
            
            # remove matching event before we start searching again
            if event_match in events:
                events.remove(event_match)
    return paired_events


def flit_crit_path(events, log_dir):
    # label event names
    gt_comp_nm      = 'Baseline Compile'
    gt_test_nm      = 'Run Test Baseline'
    comp_nm         = 'Compile'
    link_nm         = 'Linking'
    testrun_nm      = 'Run Test'
    testcompare_nm  = 'Compare Test'
    # This is not identified in log file,
    # must be manually identified by comparing compilation.
    gt_link_nm      = 'Baseline Link'

    grouped_events = dict()

    # group events by name
    for event in events:
        if event['name'] not in grouped_events:
            grouped_events[event['name']] = list()
        grouped_events[event['name']].append(event)

    # Identify baseline link vs. other linking
    gt_comp = grouped_events[gt_comp_nm][0]['message']['Compilation']
    grouped_events[gt_link_nm] = list()
    gt_links = list()
    for idx, event in enumerate(grouped_events[link_nm]):
        if event['message']['Compilation'].strip() == gt_comp.strip():
            grouped_events[gt_link_nm].append(event)
            gt_links.append(idx)
    # Remove the baseline linking step from the test compilation links
    grouped_events[link_nm] = [val for idx, val in enumerate(grouped_events[link_nm]) if idx not in gt_links]

    # Pair events with key: event_name, val: {message, time}
    paired_events = pair_events(grouped_events)

    #------------------------
    # Build map from objects to integers
    #------------------------
    # Empty map
    alias_map = { # Each entry is map of string to 0 indexed integer
                'files': dict(),
                'compilations': dict(),
                'tests': dict()
            }

    # Map files and baseline compilation
    for event in paired_events[gt_comp_nm]:
        filename = event['message']['File'].strip()
        compilation = event['message']['Compilation'].strip()

        if filename not in alias_map['files'].keys():
            alias = len(alias_map['files'])
            alias_map['files'][filename] = alias

        if compilation not in alias_map['compilations'].keys():
            alias = len(alias_map['compilations'])
            alias_map['compilations'][compilation] = alias

    # Map all other compilations
    for event in paired_events[comp_nm]:
        filename = event['message']['File'].strip()
        compilation = event['message']['Compilation'].strip()

        if filename not in alias_map['files'].keys():
            alias = len(alias_map['files'])
            alias_map['files'][filename] = alias

        if compilation not in alias_map['compilations'].keys():
            alias = len(alias_map['compilations'])
            alias_map['compilations'][compilation] = alias

    # Map tests
    for event in paired_events[gt_test_nm]:
        test = event['message']['Test'].strip()

        if test not in alias_map['tests'].keys():
            alias = len(alias_map['tests'])
            alias_map['tests'][test] = alias



    ##########################
    # Create data structure
    ##########################

    #------------------------
    # Build empty structure for graph traversal
    #------------------------
    section_template = {
                'all': None, # To hold list of times
                'crit': -1   # To hold index of critical item
            }

    test_template = {
                'compile': section_template.copy(),
                'link': section_template.copy(),
                'test': section_template.copy(),
                'compare': section_template.copy()
            }
    # Baseline data
    gt_tests = json.loads(json.dumps(test_template))
    # Test data
    individual_tests = dict()
    # All data
    graph_dict = {
                'gt_tests': gt_tests,
                'tests': individual_tests
            }

    #------------------------
    #------------------------
    # Populate data
    #------------------------
    #------------------------

    #------------------------
    # Baseline
    #------------------------
    graph_dict['gt_tests']['compile']['all'] = [-1]*len(alias_map['files'])
    graph_dict['gt_tests']['link']['all'] = [-1]
    graph_dict['gt_tests']['test']['all'] = [-1]*len(alias_map['tests'])

    # Compilation
    crit_time = 0
    for event in paired_events[gt_comp_nm]:
        filename = event['message']['File'].strip()
        file_alias = alias_map['files'][filename]

        graph_dict['gt_tests']['compile']['all'][file_alias] = event['time']

        if event['time'] > crit_time:
            crit_time = event['time']
            graph_dict['gt_tests']['compile']['crit'] = file_alias

    # Need to identify the baseline link step separately
    #graph_dict['gt_tests']['link']['all'][0] = paired_events[gt_link_nm]['time']
    
    # Test runs
    crit_time = 0
    for event in paired_events[gt_test_nm]:
        test = event['message']['Test'].strip()
        test_alias = alias_map['tests'][test]

        graph_dict['gt_tests']['test']['all'][test_alias] = event['time']

        if event['time'] > crit_time:
            crit_time = event['time']
            graph_dict['gt_tests']['test']['crit'] = file_alias


    #------------------------
    # Other compilations
    #------------------------
    # Compilation
    total_crit_time = 0
    graph_dict['tests']['crit'] = -1
    graph_dict['tests']['all'] = dict()
    for comp, alias in alias_map['compilations'].items():
        compilation = comp.strip()
        graph_dict['tests']['all'][alias] = json.loads(json.dumps(test_template))

        graph_dict['tests']['all'][alias]['compile']['all'] = [-1]*len(alias_map['files'])
        graph_dict['tests']['all'][alias]['link']['all'] = [-1]
        graph_dict['tests']['all'][alias]['test']['all'] = [-1]*len(alias_map['tests'])
        graph_dict['tests']['all'][alias]['compare']['all'] = [-1]*len(alias_map['tests'])

        total_comp_time = 0

        crit_time = 0
        for event in paired_events[comp_nm]:
            if event['message']['Compilation'].strip() != compilation:
                continue

            filename = event['message']['File'].strip()
            file_alias = alias_map['files'][filename]

            graph_dict['tests']['all'][alias]['compile']['all'][file_alias] = event['time']

            if event['time'] > crit_time:
                crit_time = event['time']
                graph_dict['tests']['all'][alias]['compile']['crit'] = file_alias
        total_comp_time += crit_time

        # Linking
        crit_time = 0
        for event in paired_events[link_nm]:
            if event['message']['Compilation'].strip() != compilation:
                continue

            graph_dict['tests']['all'][alias]['link']['all'][0] = event['time']
        total_comp_time += crit_time

        # Test Runs
        crit_time = 0
        for event in paired_events[testrun_nm]:
            if event['message']['Compilation'].strip() != compilation:
                continue

            test = event['message']['test'].strip()
            test_alias = alias_map['tests'][test]

            graph_dict['tests']['all'][alias]['test']['all'][test_alias] = event['time']

            if event['time'] > crit_time:
                crit_time = event['time']
                graph_dict['tests']['all'][alias]['test']['crit'] = test_alias
        total_comp_time += crit_time

        # Test Compare
        crit_time = 0
        for event in paired_events[testcompare_nm]:
            if event['message']['Compilation'].strip() != compilation:
                continue

            test = event['message']['test'].strip()
            test_alias = alias_map['tests'][test]

            graph_dict['tests']['all'][alias]['compare']['all'][test_alias] = event['time']

            if event['time'] > crit_time:
                crit_time = event['time']
                graph_dict['tests']['all'][alias]['compare']['crit'] = test_alias
        total_comp_time += crit_time
        
        # Track longest overall compilation
        if total_comp_time > total_crit_time:
            total_crit_time  = total_comp_time
            graph_dict['tests']['crit'] = alias

    ##########################
    # Create Graph - Full
    ##########################
    graph = pgv.AGraph(name='root', directed=True)
    graph.add_node('Start', shape='Mdiamond')
    graph.add_node('End', shape='Mdiamond')

    redLabel = {'color': 'red', 'penwidth': 8}
    blackLabel = {'penwidth': 8}

    # Create baseline cluster 
    subg = list()
    graph.add_node('Compile Baseline')
    subg.append('Compile Baseline')
    graph.add_edge('Start', 'Compile Baseline', **redLabel)

    graph.add_node('Baseline Link')
    subg.append('Baseline Link')

    #------------------------
    # Baseline path
    #------------------------
    ## File compilation
    files = graph_dict['gt_tests']['compile']['all']
    crit = graph_dict['gt_tests']['compile']['crit']
    for idx, time in enumerate(files):
        name = 'File' + str(idx) + '-GT' # + str(time)
        graph.add_node(name)
        subg.append(name)
        if idx == crit:
            graph.add_edge('Compile Baseline', name, **redLabel)
            graph.add_edge(name, 'Baseline Link', **redLabel)
        else:
            graph.add_edge('Compile Baseline', name)
            graph.add_edge(name, 'Baseline Link')
            
    ## Linking
    graph.add_node('Run Baseline Tests')
    subg.append('Run Baseline Tests')
    graph.add_edge('Baseline Link', 'Run Baseline Tests', **redLabel)

    graph.add_node('Baseline Done')
    subg.append('Baseline Done')

    ## Test Run
    tests = graph_dict['gt_tests']['test']['all']
    crit = graph_dict['gt_tests']['test']['crit']
    for idx, time in enumerate(tests):
        name = '"Test' + str(idx) + '-GT"'
        graph.add_node(name)
        subg.append(name)
        if idx == crit:
            graph.add_edge('Run Baseline Tests', name, **redLabel)
            graph.add_edge(name, 'Baseline Done', **redLabel)
        else:
            graph.add_edge('Run Baseline Tests', name)
            graph.add_edge(name, 'Baseline Done')

    graph.add_node('Test Compilations')
    graph.add_edge('Baseline Done', 'Test Compilations', **redLabel)

    graph.add_subgraph(subg, name='cluster_0', label='Run Baseline Tests', labelloc='b', labeljust='l')

    #------------------------
    # Test compilations
    #------------------------

    # Loop through compilations, creating clusters for each
    crit_id = graph_dict['tests']['crit']
    for comp_id, comp_data in graph_dict['tests']['all'].items():
        subg = list()
        comp_name = 'Compilation ' + str(comp_id)
        link_node = comp_name + ' Link'
        graph.add_node(comp_name)
        graph.add_node(link_node)
        subg.append(comp_name)
        subg.append(link_node)

        # highlight the full critical test run
        if comp_id == crit_id:
            graph.add_edge('Test Compilations', comp_name, **redLabel)
            testLabel = redLabel
        else:
            graph.add_edge('Test Compilations', comp_name)
            testLabel = blackLabel

    
        #------------------------
        # Test compilation comp_id
        #------------------------

        ## File compilation
        files = graph_dict['tests']['all'][comp_id]['compile']['all']
        crit = graph_dict['tests']['all'][comp_id]['compile']['crit']
        for idx, time in enumerate(files):
            name = 'File' + str(idx) + '-Comp' + str(comp_id)
            graph.add_node(name)
            subg.append(name)
            if idx == crit:
                graph.add_edge(comp_name, name, **testLabel)
                graph.add_edge(name, link_node, **testLabel)
            else:
                graph.add_edge(comp_name, name)
                graph.add_edge(name, link_node)
                
        ## Linking
        test_node = 'Run Tests - Compilation ' + str(comp_id)
        graph.add_node(test_node)
        subg.append(test_node)
        graph.add_edge(link_node, test_node, **testLabel)

        compare_node = 'Compare Compilation ' + str(comp_id)
        graph.add_node(compare_node)
        subg.append(compare_node)

        ## Test Run
        tests = graph_dict['tests']['all'][comp_id]['test']['all']
        crit = graph_dict['tests']['all'][comp_id]['test']['crit']
        for idx, time in enumerate(tests):
            name = 'Test' + str(idx) + '-Comp' + str(comp_id)
            graph.add_node(name)
            subg.append(name)
            if idx == crit:
                graph.add_edge(test_node, name, **testLabel)
                graph.add_edge(name, compare_node, **testLabel)
            else:
                graph.add_edge(test_node, name)
                graph.add_edge(name, compare_node)

        ## Compare Tests
        done_node = 'Compilation ' + str(comp_id) + ' done'
        graph.add_node(done_node)
        subg.append(done_node)

        tests = graph_dict['tests']['all'][comp_id]['compare']['all']
        crit = graph_dict['tests']['all'][comp_id]['compare']['crit']
        for idx, time in enumerate(tests):
            name = 'Compare Test' + str(idx) + '-Comp' + str(comp_id)
            graph.add_node(name)
            subg.append(name)
            if idx == crit:
                graph.add_edge(compare_node, name, **testLabel)
                graph.add_edge(name, done_node, **testLabel)
            else:
                graph.add_edge(compare_node, name)
                graph.add_edge(name, done_node)

        ## End
        if comp_id == crit_id:
            graph.add_edge(done_node, 'End', **redLabel)
        else:
            graph.add_edge(done_node, 'End')

        ## Cluster 0 is baseline, so all test compilation clusters start from 1
        cluster_name = 'cluster_' + str(comp_id+1)
        label_name = 'Test Compilation ' + str(comp_id)

        graph.add_subgraph(subg, cluster_name, label=label_name, labelloc='b', labeljust='l')
        
    with util.pushd(log_dir):
        graph.write('full-graph.dot')
        graph.draw('full-graph.png', prog='dot')

        with open('graph_map.txt', 'w') as fout:
            fout.write(json.dumps(alias_map))

    return graph_dict

    ##########################
    # Create Graph - Critical
    ##########################

#    redlabel = '[color="red",penwidth=8.0]'
#    blacklabel = '[penwidth=8.0]'
#
#    graph = '''
#        digraph G {
#            start [shape = Mdiamond];
#            end [shape = mDiamond];
#
#            start -> compileGT [color="red",penwidth=8.0];
#    '''
#    graph_end = ''
#
#    # Create baseline cluster
#    subgraph = '''
#        subgraph cluster_0 {
#    	    node [style=filled];
#
#            label = "Run Baseline Tests";
#            labeljust = "l";
#            labelloc = "b";
#
#    '''
#
#    ## File compilation
#    files = graph_dict['gt_tests']['compile']['all']
#    crit = graph_dict['gt_tests']['compile']['crit']
#    for idx, time in enumerate(files):
#        name = '"File' + str(idx) + '-GT"'
#        if idx == crit:
#            subgraph += 'compileGT -> ' + name + ' ' + redlabel + ';\n'
#            subgraph += name + ' -> linkGT' + redlabel + ';\n'
#        else:
#            subgraph += 'compileGT -> ' + name + ' -> linkGT;\n'
#            
#    ## Linking
#    subgraph += '\nlinksGT -> runTestsGT ' + redlabel + ';\n'
#
#    ## Test Run
#    tests = graph_dict['gt_tests']['test']['all']
#    crit = graph_dict['gt_tests']['test']['crit']
#    for idx, time in enumerate(tests):
#        name = '"Test' + str(idx) + '-GT"'
#        if idx == crit:
#            subgraph += 'runTestsGT -> ' + name + ' ' + redlabel + ';\n'
#            subgraph += name + ' -> baselineDone' + redlabel + ';\n'
#        else:
#            subgraph += 'runTestsGT -> ' + name + ' -> baselineDone;\n'
#    
#    graph_end += 'baselineDone -> testComps ' + redlabel + ';\n'
#
#    # Loop through compilations, creating clusters for each
#
#
#
#
#    return crit_events, event_times


def bisect_crit_path(events, log_dir):
    # label event names
    gt_comp_nm      = 'Compile'
    link_nm         = 'Bisect Link'
    file_bisect     = 'Bisect File'
    symbol_bisect   = 'Bisect Symbol'
    # This is not identified in log file,
    # must be manually identified by comparing compilation.
    gt_link_nm      = 'Bisect Baseline Link'


    grouped_events = dict()

    # group events by name
    for event in events:
        if event['name'] not in grouped_events:
            grouped_events[event['name']] = list()
        grouped_events[event['name']].append(event)

    # Pair events with key: event_name, val: {message, time}
    paired_events = pair_events(grouped_events)

    #------------------------
    # Build map from objects to integers
    #------------------------
    # Empty map
    alias_map = { # Each entry is map of string to 0 indexed integer
                'files': dict(),
            }

    # Map files and baseline compilation
    for event in paired_events[gt_comp_nm]:
        filename = event['message']['File'].strip()

        if filename not in alias_map['files'].keys():
            alias = len(alias_map['files'])
            alias_map['files'][filename] = alias

    ##########################
    # Create data structure
    ##########################

    #------------------------
    # Build empty structure for graph traversal
    #------------------------
    # All data
    graph_dict = {
                'gt': dict(),
                file_bisect: dict(),
                symbol_bisect: dict()
            }

    #------------------------
    #------------------------
    # Populate data
    #------------------------
    #------------------------

    #------------------------
    # Baseline
    #------------------------
    graph_dict['gt']['all'] = [-1]*len(alias_map['files'])

    # Baseline Compilation
    crit_time = 0
    for event in paired_events[gt_comp_nm]:
        # Baseline compilations should move object files to
        # the gt folder
        if event['message']['Object File'][:2] != 'gt':
            continue

        filename = event['message']['File'].strip()
        file_alias = alias_map['files'][filename]

        graph_dict['gt']['all'][file_alias] = event['time']

        if event['time'] > crit_time:
            crit_time = event['time']
            graph_dict['gt']['crit'] = file_alias
    
    # File Bisect
    for event in paired_events[file_bisect]:
        path = event['message']['Path']
        bisect_num = path[path.index('bisect-')+8:]
        graph_dict[file_bisect][bisect_num] = event['time']

    # Symbol Bisect
    graph_dict[symbol_bisect]['all'] = dict()
    crit_time = 0
    for event in paired_events[symbol_bisect]:
        path = event['message']['Path']
        bisect_num = path[path.index('bisect-')+8:]
        graph_dict[symbol_bisect]['all'][bisect_num] = event['time']

        if event['time'] > crit_time:
            crit_time = event['time']
            graph_dict[symbol_bisect]['crit'] = bisect_num

    ##########################
    # Create Graph - Full
    ##########################
    graph = pgv.AGraph(name='root', directed=True)
    graph.add_node('Start', shape='Mdiamond')
    graph.add_node('End', shape='Mdiamond')

    redLabel = {'color': 'red', 'penwidth': 8}
    blackLabel = {'penwidth': 8}

    # Create baseline cluster 
    subg = list()
    graph.add_node('Compile Baseline')
    subg.append('Compile Baseline')
    graph.add_edge('Start', 'Compile Baseline', **redLabel)

    graph.add_node('Baseline Link')
    subg.append('Baseline Link')

    graph.add_node('Baseline Done')
    subg.append('Baseline Done')

    #------------------------
    # Baseline path
    #------------------------
    ## File compilation
    files = graph_dict['gt']['compile']['all']
    crit = graph_dict['gt']['compile']['crit']
    for idx, time in enumerate(files):
        name = 'File' + str(idx) + '-GT ' + str(time)
        graph.add_node(name)
        subg.append(name)
        if idx == crit:
            graph.add_edge('Compile Baseline', name, **redLabel)
            graph.add_edge(name, 'Baseline Link', **redLabel)
        else:
            graph.add_edge('Compile Baseline', name)
            graph.add_edge(name, 'Baseline Link')
            
    ## Linking
    graph.add_edge('Baseline Link', 'Baseline Done', **redLabel)

    graph.add_node('File Bisect')
    graph.add_edge('Baseline Done', 'File Bisect', **redLabel)

    graph.add_subgraph(subg, name='cluster_0', label='Baseline Compilation', labelloc='b', labeljust='l')

    #------------------------
    # File Bisect
    #------------------------
    bisect_passes = graph_dict[file_bisect]
    subg = list()

    prev_node = 'File Bisect'
    for idx, time in bisect_passes.items():
        name = 'File Bisect - Pass ' + str(idx)
        graph.add_node(name)
        subg.append(name)

        graph.add_edge(prev_node, name, **redLabel)
        prev_node = name

    graph.add_subgraph(subg, name='cluster_1', label='File Bisect', labelloc='b', labeljust='l') 

    graph.add_node('Symbol Bisect')
    graph.add_edge(prev_node, 'Symbol Bisect', **redLabel)
    
    #------------------------
    # Symbol Bisect
    #------------------------
    files = graph_dict[symbol_bisect]['all']
    crit_file = graph_dict[symbol_bisect]['crit']
    cluster_id = str(2)

    for file_id, passes in files.items():
        subg = list()
        node_prefix = 'Symbol Bisect - File ' + str(file_id)
        prev_node = 'Symbol Bisect'

        if file_id == crit_file:
            fileLabel = redLabel
        else:
            fileLabel = blackLabel

        for pass_num, time in passes.items:
            node = node_prefix + ' Pass ' + str(pass_num)
            graph.add_node(node)
            subg.append(node)

            graph.add_edge(prev_node, node, **fileLabel)
            prev_node = node

        doneNode = node_prefix + ' Done'
        graph.add_node(doneNode)
        subg.append(doneNode)
        graph.add_edge(prev_node, doneNode, **fileLabel)

        graph.add_subgraph(subg, name='cluster_' + cluster_id, label='Symbol Bisect' + str(file_id), labelloc='b', labeljust='l')
        cluster_id += 1

        if file_id == crit_file:
            graph.add_edge(doneNode, 'End', **redLabel)
        else:
            graph.add_edge(doneNode, 'End')

    with util.pushd(log_dir):
        graph.write('bisect-full-graph.dot')
        graph.draw('bisect-full-graph.png', prog='dot')

        with open('bisect-graph_map.txt', 'w') as fout:
            fout.write(json.dumps(alias_map))

    return bisect_graph_dict


def main(arguments, prog=None):
    'Main logic here'
    parser = populate_parser()
    if prog: parser.prog = prog
    args = parser.parse_args(arguments)

    # TODO: plotting, for each event need different test cases to compare
    
    log_dir = os.path.join(args.directory, 'event_logs')

    flit_events, bisect_events = log_to_dict(log_dir)

    if args.runtype == 'flit':
        flit_crit_path(flit_events, log_dir)
    if args.runtype == 'bisect':
        bisect_crit_path(bisect_events, log_dir)

    #print critical path for now
    # crit_events, event_times = crit_path(events)
    # print('Critical events:')
    # for event in crit_events:
    #     print('Time:', event['time'], 'Event: ', event['name'] + str(event['message'], '\n')

    # Plot total time for each log event.
    # util.logplot(event_times, log_dir)

    # Temp for now, just print the results to console...
    # for key, val in event_times.items():
    #     print('Event: {}, Elapsed: {}, Dict: {}'.format(key, 
    #           val['stop_total']-val['start_total'], json.dumps(val)))

    # Create text tree of critical path
    # util.texttree(event_times, log_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
