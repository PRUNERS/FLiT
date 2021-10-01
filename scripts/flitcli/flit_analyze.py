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
import copy
import glob
import json
import os
import sys
import importlib
from operator import and_
from functools import reduce
from io import StringIO
from datetime import timedelta, datetime
from collections import defaultdict

# NetworkX is a graph utility for managing the data structure.
# https://networkx.org/documentation/stable/index.html
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv

import flitutil as util

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
    parser.add_argument('--definitions', default = {},
                        help='Path to python file with dictionary of event definitions.')    
    parser.add_argument('-l', '--logs', default=False, action='store_true',
                        help='Enable logging of FLiT events.')
    parser.add_argument('-r', '--runtype', default='flit',
                        help='''
                        Type of run to analyze. Options are "flit" or
                        "bisect".
                        ''')

    return parser


def import_dependencies(filepath):
    
    
    return dependency_dict


class Event:
    '''
    An event object.

    Public attributes:
    - name: the name of the event
    - type: START, STOP, or DURATION
    - nanosecs_since_epoch: a time point for the event.  For DURATION events,
      this marks the beginning of the event.
    - datetime: a time point for the event.  For DURATION events, this marks
      the beginning of the event.
    - properties (dict): detailed information specific to the type of event
    - duration: only useful for DURATION events, nanosecond duration
    - children (list): children events (only for DURATION type)
    - parent: None if no parent, else another Event object
    '''

    START = 1
    STOP = 2
    DURATION = 3
    
    __eventCount = 0

    # This dictionary should be provided in a Python file
    # passed as an argument when calling this function.
    event_dependencies = None
    
    def __init__(self, values=None):
        self._id = Event.__eventCount
        Event.__eventCount += 1
        
        self.children = [] # a list of Event objects
        self.nested_children = [] # a list of Event objects

        self.nested_parent = None # one other Event object
        self.parents = [] # a list of Event objects

        if values:
            self.populate(values)
        else:
            self._name = ''
            self._label = ''
            self._type = Event.DURATION
            self._nanosecs_since_epoch = 0
            self._datetime = datetime(1970, 1, 1)
            self._properties = {}
            self._duration = 0

    def populate(self, values):
        self.name = values['name']
        self._type = Event.START if values['type'] == 'start' else Event.STOP
        self._nanosecs_since_epoch = values['time']
        self._datetime = \
            datetime(1970, 1, 1) + \
            timedelta(milliseconds=self.nanosecs_since_epoch // 1000000)
        self._properties = values['properties']
        self._duration = 0


    @property
    def id(self):
        return self._id

   
    @property
    def name(self):
        return self._name
   
   
    @name.setter
    def name(self, newName):
        self._name = newName
        self._label = str(self._id) + ' - ' + newName

    
    @property
    def label(self):
        return self._label


    @property
    def type(self):
        return self._type
   
   
    @type.setter
    def type(self, newType):
        self._type = newType


    @property
    def nanosecs_since_epoch(self):
        return self._nanosecs_since_epoch
   
   
    @nanosecs_since_epoch.setter
    def nanosecs_since_epoch(self, newNano):
        self._nanosecs_since_epoch = newNano


    @property
    def datetime(self):
        return self._datetime
   
   
    @datetime.setter
    def datetime(self, newDate):
        self._datetime = newDate


    @property
    def properties(self):
        return self._properties
   
   
    @properties.setter
    def properties(self, newProps):
        self._properties = newProps


    @property
    def duration(self):
        return self._duration
   
   
    @duration.setter
    def duration(self, newDuration):
        self._duration = newDuration


    @staticmethod
    def create_root_event(first_timestamp, last_timestamp):
        '''
        From two timestamps of nanoseconds since the epoch, create a duration
        event (called "root") and return it.
        '''
        root_event = Event()
        root_event.name = 'root'
        root_event.type = Event.DURATION
        root_event.duration = last_timestamp - first_timestamp
        root_event.datetime = \
            datetime(1970, 1, 1) + timedelta(milliseconds=first_timestamp // 1000000)
        root_event.nanosecs_since_epoch = first_timestamp
        root_event.properties = {}


        # create a placeholder parent for events with
        # undefined relationships
        undef = Event()
        undef.name = 'Undefined'
       
        return root_event, undef

    def connect_nested_parent(self, possible_parents):
        d = Event.event_dependencies.get(self.name, {'Nested Parent': 'Undefined', 'Parent Name': [], 'Matching': None})

        # find the matches
        matching_nested_parents = (x for x in possible_parents if x.name in d['Nested Parent'])
        matching_nested_parents = [x for x in matching_nested_parents
                                   if x.name == 'root'
                                   or x.name == 'Undefined'
                                   or d['Matching'] is None
                                   or d['Matching'](self, x)]

        # make sure we only have one nested parent
        assert len(matching_nested_parents) > 0, 'Nested parent not found: ' + self._label
        assert len(matching_nested_parents) == 1, 'More than one nested parents found: ' + self._label
        matching_nested_parent = matching_nested_parents[0]
        del matching_nested_parents

        # create connections
        self.nested_parent = matching_nested_parent
        matching_nested_parent.nested_children.append(self)

    def connect_parents(self, possible_parents):
        '''
        Attach this event to its parent with a two-way link.

        Logic depends on event: any event to be analyzed
        in the FLiT workflow needs a separate case with its
        logical dependencies defined in the event_dependencies
        dictionary.

        @param possible_parents (list(Event)): events that have turned into
            Duration events and can be used as sequential dependencies.
        '''
        d = Event.event_dependencies.get(self.name, {'Nested Parent': 'Undefined', 'Parent Name': [], 'Matching': None})

        # find the matches
        matching_parents = (x for x in possible_parents if x.name in d['Parent Name'])
        matching_parents = [x for x in matching_parents
                            if d['Matching'] is None
                            or d['Matching'](self, x)]

        # create connections
        self.parents.extend(matching_parents)
        for parent in matching_parents:
            parent.children.append(self)


    def __eq__(self, other):
        return (self.name == other.name and self.properties == other.properties)

    def __repr__(self):
        return repr(self.label)

    def __str__(self):
        return '\n'.join('{}: {}'.format(name, repr(val)) for name, val in vars(self).items())

    #def __hash__(self):
    #    '''
    #    Using Python's built in hashes for tuples and strings.
    #    '''
    #    prop_list = list(self.properties.items())
    #    prop_list.append(self.name)
    #    prop_tuple = tuple(prop_list)
    #    return hash(prop_tuple)


def parse_logs(logfiles):
    '''
    Read raw log files and aggregate to single log file.
    Returns a list of individual event dictionaries.
    '''
    events = []
    for filename in logfiles:
        with open(filename, 'r') as fin:
            events.extend(Event(json.loads(line)) for line in fin)
    events.sort(key=lambda x: x.nanosecs_since_epoch)
    return events


def gen_event_graph(events):
    '''
    Take a list of events and create a graph (by populating event.children
    and event.duration) of events.
    
    @param events: list of individual events in dict format
    
    @return networkx event graph, root node for DAG defined by event
                relationships, and undefined node connected to all events
                with undefined relationships
    '''
    events.sort(key=lambda x: x.nanosecs_since_epoch)

    root, undef = Event.create_root_event(events[0].nanosecs_since_epoch,
                                   events[-1].nanosecs_since_epoch)

    # Each node will:
    # - either belong to root or undef
    # - have exactly one nested parent (except for root and undef)
    # - have zero or more sequential parents (sequential dependency)
    #
    # Each node should:
    # - be a duration type
    # - have a start and duration
    # - have zero sequential parents only if they are the top-level tasks in a nested scope
    # - have zero sequential children only if they are the end tasks in a nested scope
    # - root and leaf nodes within a nested scope may have parents and children from other nested scopes

    # TODO: we depend here on sequential consistency.  If we move to multiple
    # TODO: hosts, then this needs to be revisited, perhaps using Lamport
    # TODO: clocks.

    # The beginning and ending events match both in name and in properties exactly.
    matched = []
    unmatched = [root, undef]

    # Group all events by name for parent identification
    event_dict = defaultdict(lambda: [])
    for event in events:
        assert event.type != Event.DURATION, "Duration event in event list! " + str(event.id)
        if event.type == Event.START:
            event_copy = copy.copy(event)
            event_copy.connect_nested_parent(unmatched)
            unmatched.append(event_copy)
        elif event.type == Event.STOP:
            # match with something in the unmatched list
            matching = [x for x in unmatched if event == x]  # everything but type is the same
            assert len(matching) > 0, "could not find a matching event: " + str(event.id)
            assert len(matching) < 2, "multiple matches for this event found: " + str(event.id)
            matching = matching[0]
            unmatched.remove(matching)

            # assert that the nesting parent has not ended yet
            assert matching.nested_parent is not None
            assert matching.nested_parent in unmatched, 'Parent cannot end before nested children'

            matching.type = Event.DURATION
            matching.duration = \
                    event.nanosecs_since_epoch - matching.nanosecs_since_epoch
            assert matching.duration >= 0, "Negative duration! " + str(matching.id)

            matching.connect_parents(matched)
            matched.append(matching)
        else:
            raise AssertionError('Event is not a START or STOP type')

    # we are now done with the top-level events
    unmatched.remove(root)
    unmatched.remove(undef)
    matched.append(root)
    matched.append(undef)

    # make sure no straglers are still there
    assert len(unmatched) == 0

    # TODO: assert all nodes are DURATION nodes
    # sample idea: func = lambda l: reduce(and_,[a == 1 for a in l])
    # broken for now.
    # all_duration = lambda root: reduce(and_, [(root.type == Event.DURATION and all_duration(e)) for e in root.children])
    # assert all_duration(root)

    # For now, we create the graph at the very end (after we have created our
    # graph with our Event data structure
    # TODO: separate out this logic into two separate functions
    # TODO: define edge weights as durations
    # TODO: Update parent/child relationships with graph connections.
    G = nx.DiGraph()
    edges = []
    for event in matched:
        G.add_node(event.label)
        G.nodes[event.label]['event'] = event
        if event.nested_parent is not None:
            edges.append((
                event.nested_parent.label,
                event.label,
                event.nanosecs_since_epoch - event.nested_parent.nanosecs_since_epoch
                ))
        edges.extend((p.label, event.label, event.duration) for p in event.parents)

    # root is only connected by nesting relationship, so need edges here.
    edges.extend((root.label, node.label, node.duration) for node in root.nested_children)
    
    G.add_weighted_edges_from(edges)

    # TODO: also return matched list
    return G, root, undef


#def tree_toString(root, indent=0, outString=''):
#    '''
#    Returns a string describing tree structure of events from root
#    '''
#    prefix = ' '*indent + '|--'
#    line = prefix + root.name + ' || ' + str(root.duration) + \
#            'ns || ' + str(root.properties) + '\n'
#    #line = spaces + 'Event: ' + root.name + ' || Duration: ' + str(root.duration) \
#    #        + 'ns' + ' || Properties: ' + str(root.properties)[:50] + '\n'
#    outString += line
#
#    for node in root.children:
#        outString += tree_toString(node, indent+2)
#
#    return outString


def main(arguments, prog=None):
    'Main logic here'
    parser = populate_parser()
    if prog: parser.prog = prog
    args = parser.parse_args(arguments)

    # TODO: plotting, for each event need different test cases to compare
    # TODO: fix log_dir setup; grab from toml?
    # TODO: implement hierarchy parent/child relationship
    # TODO: decide what to do about disabled tests
    
    # TODO: Very hacky, this import process needs to be made 
    #   more robust and general
    # Get event dependencies from user-provided file
    definition_file = args.definitions
    
    def_path = os.path.dirname(os.path.abspath(definition_file))
    if def_path not in sys.path:
        sys.path.insert(0, def_path)
    
    def_module = importlib.import_module(os.path.basename(definition_file))
    Event.event_dependencies = def_module.event_dependencies
    
    # Read logfiles from user-provided directory
    log_dir = os.path.join(args.directory, 'event_logs')

    with util.pushd(log_dir):
        logfiles = glob.glob('*.log')
        flit_events = parse_logs(logfiles)

    event_graph, root, undef = gen_event_graph(flit_events)
    
    #nx.draw(G)
    #plt.savefig('plot.png')
    
    #print(tree_toString(event_root))

    #if args.runtype == 'flit':
    #    flit_crit_path(flit_events, log_dir)
    #if args.runtype == 'bisect':
    #    bisect_crit_path(bisect_events, log_dir)

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



##
## OLD STUFF BELOW
##

#def get_event_duration(events):
#    '''
#    Aggregates total time accumulated for each event name.
#    Returns a dictionary of event names with aggregated data.
#    '''
#    event_times = {}
#
#    for event in events:
#        name = event['name']
#        if name not in event_times:
#            event_times[name] = {'start_total': 0, 'stop_total': 0,
#                                 'start_count': 0, 'stop_count': 0}
#        if event['start_stop'] == 'start':
#            event_times[name]['start_total'] += event['time']
#            # for error checking
#            event_times[name]['start_count'] += 1
#        else:
#            event_times[name]['stop_total'] += event['time']
#            # for error checking
#            event_times[name]['stop_count'] += 1
#   
#    # Should have equal starts and stops for each event
#    for key, val in event_times.items():
#        assert val['start_count'] == val['stop_count'], \
#               'Unequal start/stop count for event: ' + key + \
#               ' starts: ' + str(val['start_count']) + ' stops: ' + str(val['stop_count'])
#        assert val['start_total'] < val['stop_total'], \
#               'Start time should be less than stop time for event: ' + key
# 
#    return event_times
#
#
#def pair_events(grouped_events):
#    '''
#        Takes in a dict of grouped events
#        Returns dict after combining start/stop pairs
#    '''
#    paired_events = dict()
#
#    # go through each event type (name)
#    for event_name, v in grouped_events.items():
#        events = list(v)
#
#        # for each individual event of this type, find its start/stop pairing
#        while len(events) > 0:
#            this_event = events.pop()
#            event_match = None
#            # go through events of this name to find matching message (i.e. find pair)
#            for event in events:
#                if event['message'] == this_event['message']:
#                    event_match = event
#                    time = abs(event_match['time'] - this_event['time'])
#                    # keep track of time for each event
#                    if event_name.strip() not in paired_events:
#                        paired_events[event_name.strip()] = list()
#                    paired_events[event_name.strip()].append({'message':this_event['message'], 'time':time})
#            
#            # remove matching event before we start searching again
#            if event_match in events:
#                events.remove(event_match)
#    return paired_events
#
#
#def flit_crit_path(events, log_dir):
#    # label event names
#    gt_comp_nm      = 'Baseline Compile'
#    gt_test_nm      = 'Run Test Baseline'
#    comp_nm         = 'Compile'
#    link_nm         = 'Linking'
#    testrun_nm      = 'Run Test'
#    testcompare_nm  = 'Compare Test'
#    # This is not identified in log file,
#    # must be manually identified by comparing compilation.
#    gt_link_nm      = 'Baseline Link'
#
#    grouped_events = dict()
#
#    # group events by name
#    for event in events:
#        if event['name'] not in grouped_events:
#            grouped_events[event['name']] = list()
#        grouped_events[event['name']].append(event)
#
#    # Identify baseline link vs. other linking
#    gt_comp = grouped_events[gt_comp_nm][0]['message']['Compilation']
#    grouped_events[gt_link_nm] = list()
#    gt_links = list()
#    for idx, event in enumerate(grouped_events[link_nm]):
#        if event['message']['Compilation'].strip() == gt_comp.strip():
#            grouped_events[gt_link_nm].append(event)
#            gt_links.append(idx)
#    # Remove the baseline linking step from the test compilation links
#    for event_index in gt_links:
#        grouped_events[link_nm].pop(event_index)
#
#    # Pair events with key: event_name, val: {message, time}
#    paired_events = pair_events(grouped_events)
#
#    #------------------------
#    # Build map from objects to integers
#    #------------------------
#    # Empty map
#    alias_map = { # Each entry is map of string to 0 indexed integer
#                'files': dict(),
#                'compilations': dict(),
#                'tests': dict()
#            }
#
#    # Map files and baseline compilation
#    for event in paired_events[gt_comp_nm]:
#        filename = event['message']['File'].strip()
#        compilation = event['message']['Compilation'].strip()
#
#        if filename not in alias_map['files'].keys():
#            alias = len(alias_map['files'])
#            alias_map['files'][filename] = alias
#
#        if compilation not in alias_map['compilations'].keys():
#            alias = len(alias_map['compilations'])
#            alias_map['compilations'][compilation] = alias
#
#    # Map all other compilations
#    for event in paired_events[comp_nm]:
#        filename = event['message']['File'].strip()
#        compilation = event['message']['Compilation'].strip()
#
#        if filename not in alias_map['files'].keys():
#            alias = len(alias_map['files'])
#            alias_map['files'][filename] = alias
#
#        if compilation not in alias_map['compilations'].keys():
#            alias = len(alias_map['compilations'])
#            alias_map['compilations'][compilation] = alias
#
#    # Map tests
#    for event in paired_events[gt_test_nm]:
#        test = event['message']['Test'].strip()
#
#        if test not in alias_map['tests'].keys():
#            alias = len(alias_map['tests'])
#            alias_map['tests'][test] = alias
#
#
#
#    ##########################
#    # Create data structure
#    ##########################
#
#    #------------------------
#    # Build empty structure for graph traversal
#    #------------------------
#    section_template = {
#                'all': None, # To hold list of times
#                'crit': -1   # To hold index of critical item
#            }
#
#    test_template = {
#                'compile': section_template.copy(),
#                'link': section_template.copy(),
#                'test': section_template.copy(),
#                'compare': section_template.copy()
#            }
#    # Baseline data
#    gt_tests = json.loads(json.dumps(test_template))
#    # Test data
#    individual_tests = dict()
#    # All data
#    graph_dict = {
#                'gt_tests': gt_tests,
#                'tests': individual_tests
#            }
#
#    #------------------------
#    #------------------------
#    # Populate data
#    #------------------------
#    #------------------------
#
#    #------------------------
#    # Baseline
#    #------------------------
#    graph_dict['gt_tests']['compile']['all'] = [-1]*len(alias_map['files'])
#    graph_dict['gt_tests']['link']['all'] = [-1]
#    graph_dict['gt_tests']['test']['all'] = [-1]*len(alias_map['tests'])
#
#    # Compilation
#    crit_time = 0
#    for event in paired_events[gt_comp_nm]:
#        filename = event['message']['File'].strip()
#        file_alias = alias_map['files'][filename]
#
#        graph_dict['gt_tests']['compile']['all'][file_alias] = event['time']
#
#        if event['time'] > crit_time:
#            crit_time = event['time']
#            graph_dict['gt_tests']['compile']['crit'] = file_alias
#
#    # Need to identify the baseline link step separately
#    #graph_dict['gt_tests']['link']['all'][0] = paired_events[gt_link_nm]['time']
#    
#    # Test runs
#    crit_time = 0
#    for event in paired_events[gt_test_nm]:
#        test = event['message']['Test'].strip()
#        test_alias = alias_map['tests'][test]
#
#        graph_dict['gt_tests']['test']['all'][test_alias] = event['time']
#
#        if event['time'] > crit_time:
#            crit_time = event['time']
#            graph_dict['gt_tests']['test']['crit'] = file_alias
#
#
#    #------------------------
#    # Other compilations
#    #------------------------
#    # Compilation
#    total_crit_time = 0
#    graph_dict['tests']['crit'] = -1
#    graph_dict['tests']['all'] = dict()
#    for comp, alias in alias_map['compilations'].items():
#        compilation = comp.strip()
#        graph_dict['tests']['all'][alias] = json.loads(json.dumps(test_template))
#
#        graph_dict['tests']['all'][alias]['compile']['all'] = [-1]*len(alias_map['files'])
#        graph_dict['tests']['all'][alias]['link']['all'] = [-1]
#        graph_dict['tests']['all'][alias]['test']['all'] = [-1]*len(alias_map['tests'])
#        graph_dict['tests']['all'][alias]['compare']['all'] = [-1]*len(alias_map['tests'])
#
#        total_comp_time = 0
#
#        crit_time = 0
#        for event in paired_events[comp_nm]:
#            if event['message']['Compilation'].strip() != compilation:
#                continue
#
#            filename = event['message']['File'].strip()
#            file_alias = alias_map['files'][filename]
#
#            graph_dict['tests']['all'][alias]['compile']['all'][file_alias] = event['time']
#
#            if event['time'] > crit_time:
#                crit_time = event['time']
#                graph_dict['tests']['all'][alias]['compile']['crit'] = file_alias
#        total_comp_time += crit_time
#
#        # Linking
#        crit_time = 0
#        for event in paired_events[link_nm]:
#            if event['message']['Compilation'].strip() != compilation:
#                continue
#
#            graph_dict['tests']['all'][alias]['link']['all'][0] = event['time']
#        total_comp_time += crit_time
#
#        # Test Runs
#        crit_time = 0
#        for event in paired_events[testrun_nm]:
#            if event['message']['Compilation'].strip() != compilation:
#                continue
#
#            test = event['message']['test'].strip()
#            test_alias = alias_map['tests'][test]
#
#            graph_dict['tests']['all'][alias]['test']['all'][test_alias] = event['time']
#
#            if event['time'] > crit_time:
#                crit_time = event['time']
#                graph_dict['tests']['all'][alias]['test']['crit'] = test_alias
#        total_comp_time += crit_time
#
#        # Test Compare
#        crit_time = 0
#        for event in paired_events[testcompare_nm]:
#            if event['message']['Compilation'].strip() != compilation:
#                continue
#
#            test = event['message']['test'].strip()
#            test_alias = alias_map['tests'][test]
#
#            graph_dict['tests']['all'][alias]['compare']['all'][test_alias] = event['time']
#
#            if event['time'] > crit_time:
#                crit_time = event['time']
#                graph_dict['tests']['all'][alias]['compare']['crit'] = test_alias
#        total_comp_time += crit_time
#        
#        # Track longest overall compilation
#        if total_comp_time > total_crit_time:
#            total_crit_time  = total_comp_time
#            graph_dict['tests']['crit'] = alias
#
#    ##########################
#    # Create Graph - Full
#    ##########################
#    graph = pgv.AGraph(name='root', directed=True)
#    graph.add_node('Start', shape='Mdiamond')
#    graph.add_node('End', shape='Mdiamond')
#
#    redLabel = {'color': 'red', 'penwidth': 8}
#    blackLabel = {'penwidth': 8}
#
#    # Create baseline cluster 
#    subg = list()
#    graph.add_node('Compile Baseline')
#    subg.append('Compile Baseline')
#    graph.add_edge('Start', 'Compile Baseline', **redLabel)
#
#    graph.add_node('Baseline Link')
#    subg.append('Baseline Link')
#
#    #------------------------
#    # Baseline path
#    #------------------------
#    ## File compilation
#    files = graph_dict['gt_tests']['compile']['all']
#    crit = graph_dict['gt_tests']['compile']['crit']
#    for idx, time in enumerate(files):
#        name = 'File' + str(idx) + '-GT' # + str(time)
#        graph.add_node(name)
#        subg.append(name)
#        if idx == crit:
#            graph.add_edge('Compile Baseline', name, **redLabel)
#            graph.add_edge(name, 'Baseline Link', **redLabel)
#        else:
#            graph.add_edge('Compile Baseline', name)
#            graph.add_edge(name, 'Baseline Link')
#            
#    ## Linking
#    graph.add_node('Run Baseline Tests')
#    subg.append('Run Baseline Tests')
#    graph.add_edge('Baseline Link', 'Run Baseline Tests', **redLabel)
#
#    graph.add_node('Baseline Done')
#    subg.append('Baseline Done')
#
#    ## Test Run
#    tests = graph_dict['gt_tests']['test']['all']
#    crit = graph_dict['gt_tests']['test']['crit']
#    for idx, time in enumerate(tests):
#        name = '"Test' + str(idx) + '-GT"'
#        graph.add_node(name)
#        subg.append(name)
#        if idx == crit:
#            graph.add_edge('Run Baseline Tests', name, **redLabel)
#            graph.add_edge(name, 'Baseline Done', **redLabel)
#        else:
#            graph.add_edge('Run Baseline Tests', name)
#            graph.add_edge(name, 'Baseline Done')
#
#    graph.add_node('Test Compilations')
#    graph.add_edge('Baseline Done', 'Test Compilations', **redLabel)
#
#    graph.add_subgraph(subg, name='cluster_0', label='Run Baseline Tests', labelloc='b', labeljust='l')
#
#    #------------------------
#    # Test compilations
#    #------------------------
#
#    # Loop through compilations, creating clusters for each
#    crit_id = graph_dict['tests']['crit']
#    for comp_id, comp_data in graph_dict['tests']['all'].items():
#        subg = list()
#        comp_name = 'Compilation ' + str(comp_id)
#        link_node = comp_name + ' Link'
#        graph.add_node(comp_name)
#        graph.add_node(link_node)
#        subg.append(comp_name)
#        subg.append(link_node)
#
#        # highlight the full critical test run
#        if comp_id == crit_id:
#            graph.add_edge('Test Compilations', comp_name, **redLabel)
#            testLabel = redLabel
#        else:
#            graph.add_edge('Test Compilations', comp_name)
#            testLabel = blackLabel
#
#    
#        #------------------------
#        # Test compilation comp_id
#        #------------------------
#
#        ## File compilation
#        files = graph_dict['tests']['all'][comp_id]['compile']['all']
#        crit = graph_dict['tests']['all'][comp_id]['compile']['crit']
#        for idx, time in enumerate(files):
#            name = 'File' + str(idx) + '-Comp' + str(comp_id)
#            graph.add_node(name)
#            subg.append(name)
#            if idx == crit:
#                graph.add_edge(comp_name, name, **testLabel)
#                graph.add_edge(name, link_node, **testLabel)
#            else:
#                graph.add_edge(comp_name, name)
#                graph.add_edge(name, link_node)
#                
#        ## Linking
#        test_node = 'Run Tests - Compilation ' + str(comp_id)
#        graph.add_node(test_node)
#        subg.append(test_node)
#        graph.add_edge(link_node, test_node, **testLabel)
#
#        compare_node = 'Compare Compilation ' + str(comp_id)
#        graph.add_node(compare_node)
#        subg.append(compare_node)
#
#        ## Test Run
#        tests = graph_dict['tests']['all'][comp_id]['test']['all']
#        crit = graph_dict['tests']['all'][comp_id]['test']['crit']
#        for idx, time in enumerate(tests):
#            name = 'Test' + str(idx) + '-Comp' + str(comp_id)
#            graph.add_node(name)
#            subg.append(name)
#            if idx == crit:
#                graph.add_edge(test_node, name, **testLabel)
#                graph.add_edge(name, compare_node, **testLabel)
#            else:
#                graph.add_edge(test_node, name)
#                graph.add_edge(name, compare_node)
#
#        ## Compare Tests
#        done_node = 'Compilation ' + str(comp_id) + ' done'
#        graph.add_node(done_node)
#        subg.append(done_node)
#
#        tests = graph_dict['tests']['all'][comp_id]['compare']['all']
#        crit = graph_dict['tests']['all'][comp_id]['compare']['crit']
#        for idx, time in enumerate(tests):
#            name = 'Compare Test' + str(idx) + '-Comp' + str(comp_id)
#            graph.add_node(name)
#            subg.append(name)
#            if idx == crit:
#                graph.add_edge(compare_node, name, **testLabel)
#                graph.add_edge(name, done_node, **testLabel)
#            else:
#                graph.add_edge(compare_node, name)
#                graph.add_edge(name, done_node)
#
#        ## End
#        if comp_id == crit_id:
#            graph.add_edge(done_node, 'End', **redLabel)
#        else:
#            graph.add_edge(done_node, 'End')
#
#        ## Cluster 0 is baseline, so all test compilation clusters start from 1
#        cluster_name = 'cluster_' + str(comp_id+1)
#        label_name = 'Test Compilation ' + str(comp_id)
#
#        graph.add_subgraph(subg, cluster_name, label=label_name, labelloc='b', labeljust='l')
#        
#    with util.pushd(log_dir):
#        graph.write('full-graph.dot')
#        graph.draw('full-graph.png', prog='dot')
#
#        with open('graph_map.txt', 'w') as fout:
#            fout.write(json.dumps(alias_map))
#
#    return graph_dict
#
#    ##########################
#    # Create Graph - Critical
#    ##########################
#
##    redlabel = '[color="red",penwidth=8.0]'
##    blacklabel = '[penwidth=8.0]'
##
##    graph = '''
##        digraph G {
##            start [shape = Mdiamond];
##            end [shape = mDiamond];
##
##            start -> compileGT [color="red",penwidth=8.0];
##    '''
##    graph_end = ''
##
##    # Create baseline cluster
##    subgraph = '''
##        subgraph cluster_0 {
##    	    node [style=filled];
##
##            label = "Run Baseline Tests";
##            labeljust = "l";
##            labelloc = "b";
##
##    '''
##
##    ## File compilation
##    files = graph_dict['gt_tests']['compile']['all']
##    crit = graph_dict['gt_tests']['compile']['crit']
##    for idx, time in enumerate(files):
##        name = '"File' + str(idx) + '-GT"'
##        if idx == crit:
##            subgraph += 'compileGT -> ' + name + ' ' + redlabel + ';\n'
##            subgraph += name + ' -> linkGT' + redlabel + ';\n'
##        else:
##            subgraph += 'compileGT -> ' + name + ' -> linkGT;\n'
##            
##    ## Linking
##    subgraph += '\nlinksGT -> runTestsGT ' + redlabel + ';\n'
##
##    ## Test Run
##    tests = graph_dict['gt_tests']['test']['all']
##    crit = graph_dict['gt_tests']['test']['crit']
##    for idx, time in enumerate(tests):
##        name = '"Test' + str(idx) + '-GT"'
##        if idx == crit:
##            subgraph += 'runTestsGT -> ' + name + ' ' + redlabel + ';\n'
##            subgraph += name + ' -> baselineDone' + redlabel + ';\n'
##        else:
##            subgraph += 'runTestsGT -> ' + name + ' -> baselineDone;\n'
##    
##    graph_end += 'baselineDone -> testComps ' + redlabel + ';\n'
##
##    # Loop through compilations, creating clusters for each
##
##
##
##
##    return crit_events, event_times
#
#
#def bisect_crit_path(events, log_dir):
#    # label event names
#    gt_comp_nm      = 'Compile'
#    link_nm         = 'Bisect Link'
#    file_bisect     = 'Bisect File'
#    symbol_bisect   = 'Bisect Symbol'
#    # This is not identified in log file,
#    # must be manually identified by comparing compilation.
#    gt_link_nm      = 'Bisect Baseline Link'
#
#
#    grouped_events = dict()
#
#    # group events by name
#    for event in events:
#        if event['name'] not in grouped_events:
#            grouped_events[event['name']] = list()
#        grouped_events[event['name']].append(event)
#
#    # Pair events with key: event_name, val: {message, time}
#    paired_events = pair_events(grouped_events)
#
#    #------------------------
#    # Build map from objects to integers
#    #------------------------
#    # Empty map
#    alias_map = { # Each entry is map of string to 0 indexed integer
#                'files': dict(),
#            }
#
#    # Map files and baseline compilation
#    for event in paired_events[gt_comp_nm]:
#        filename = event['message']['File'].strip()
#
#        if filename not in alias_map['files'].keys():
#            alias = len(alias_map['files'])
#            alias_map['files'][filename] = alias
#
#    ##########################
#    # Create data structure
#    ##########################
#
#    #------------------------
#    # Build empty structure for graph traversal
#    #------------------------
#    # All data
#    graph_dict = {
#                'gt': dict(),
#                file_bisect: dict(),
#                symbol_bisect: dict()
#            }
#
#    #------------------------
#    #------------------------
#    # Populate data
#    #------------------------
#    #------------------------
#
#    #------------------------
#    # Baseline
#    #------------------------
#    graph_dict['gt']['all'] = [-1]*len(alias_map['files'])
#
#    # Baseline Compilation
#    crit_time = 0
#    for event in paired_events[gt_comp_nm]:
#        # Baseline compilations should move object files to
#        # the gt folder
#        if event['message']['Object File'][:2] != 'gt':
#            continue
#
#        filename = event['message']['File'].strip()
#        file_alias = alias_map['files'][filename]
#
#        graph_dict['gt']['all'][file_alias] = event['time']
#
#        if event['time'] > crit_time:
#            crit_time = event['time']
#            graph_dict['gt']['crit'] = file_alias
#    
#    # File Bisect
#    for event in paired_events[file_bisect]:
#        path = event['message']['Path']
#        bisect_num = path[path.index('bisect-')+8:]
#        graph_dict[file_bisect][bisect_num] = event['time']
#
#    # Symbol Bisect
#    graph_dict[symbol_bisect]['all'] = dict()
#    crit_time = 0
#    for event in paired_events[symbol_bisect]:
#        path = event['message']['Path']
#        bisect_num = path[path.index('bisect-')+8:]
#        graph_dict[symbol_bisect]['all'][bisect_num] = event['time']
#
#        if event['time'] > crit_time:
#            crit_time = event['time']
#            graph_dict[symbol_bisect]['crit'] = bisect_num
#
#    ##########################
#    # Create Graph - Full
#    ##########################
#    graph = pgv.AGraph(name='root', directed=True)
#    graph.add_node('Start', shape='Mdiamond')
#    graph.add_node('End', shape='Mdiamond')
#
#    redLabel = {'color': 'red', 'penwidth': 8}
#    blackLabel = {'penwidth': 8}
#
#    # Create baseline cluster 
#    subg = list()
#    graph.add_node('Compile Baseline')
#    subg.append('Compile Baseline')
#    graph.add_edge('Start', 'Compile Baseline', **redLabel)
#
#    graph.add_node('Baseline Link')
#    subg.append('Baseline Link')
#
#    graph.add_node('Baseline Done')
#    subg.append('Baseline Done')
#
#    #------------------------
#    # Baseline path
#    #------------------------
#    ## File compilation
#    files = graph_dict['gt']['compile']['all']
#    crit = graph_dict['gt']['compile']['crit']
#    for idx, time in enumerate(files):
#        name = 'File' + str(idx) + '-GT ' + str(time)
#        graph.add_node(name)
#        subg.append(name)
#        if idx == crit:
#            graph.add_edge('Compile Baseline', name, **redLabel)
#            graph.add_edge(name, 'Baseline Link', **redLabel)
#        else:
#            graph.add_edge('Compile Baseline', name)
#            graph.add_edge(name, 'Baseline Link')
#            
#    ## Linking
#    graph.add_edge('Baseline Link', 'Baseline Done', **redLabel)
#
#    graph.add_node('File Bisect')
#    graph.add_edge('Baseline Done', 'File Bisect', **redLabel)
#
#    graph.add_subgraph(subg, name='cluster_0', label='Baseline Compilation', labelloc='b', labeljust='l')
#
#    #------------------------
#    # File Bisect
#    #------------------------
#    bisect_passes = graph_dict[file_bisect]
#    subg = list()
#
#    prev_node = 'File Bisect'
#    for idx, time in bisect_passes.items():
#        name = 'File Bisect - Pass ' + str(idx)
#        graph.add_node(name)
#        subg.append(name)
#
#        graph.add_edge(prev_node, name, **redLabel)
#        prev_node = name
#
#    graph.add_subgraph(subg, name='cluster_1', label='File Bisect', labelloc='b', labeljust='l') 
#
#    graph.add_node('Symbol Bisect')
#    graph.add_edge(prev_node, 'Symbol Bisect', **redLabel)
#    
#    #------------------------
#    # Symbol Bisect
#    #------------------------
#    files = graph_dict[symbol_bisect]['all']
#    crit_file = graph_dict[symbol_bisect]['crit']
#    cluster_id = str(2)
#
#    for file_id, passes in files.items():
#        subg = list()
#        node_prefix = 'Symbol Bisect - File ' + str(file_id)
#        prev_node = 'Symbol Bisect'
#
#        if file_id == crit_file:
#            fileLabel = redLabel
#        else:
#            fileLabel = blackLabel
#
#        for pass_num, time in passes.items:
#            node = node_prefix + ' Pass ' + str(pass_num)
#            graph.add_node(node)
#            subg.append(node)
#
#            graph.add_edge(prev_node, node, **fileLabel)
#            prev_node = node
#
#        doneNode = node_prefix + ' Done'
#        graph.add_node(doneNode)
#        subg.append(doneNode)
#        graph.add_edge(prev_node, doneNode, **fileLabel)
#
#        graph.add_subgraph(subg, name='cluster_' + cluster_id, label='Symbol Bisect' + str(file_id), labelloc='b', labeljust='l')
#        cluster_id += 1
#
#        if file_id == crit_file:
#            graph.add_edge(doneNode, 'End', **redLabel)
#        else:
#            graph.add_edge(doneNode, 'End')
#
#    with util.pushd(log_dir):
#        graph.write('bisect-full-graph.dot')
#        graph.draw('bisect-full-graph.png', prog='dot')
#
#        with open('bisect-graph_map.txt', 'w') as fout:
#            fout.write(json.dumps(alias_map))
#
#    return bisect_graph_dict
