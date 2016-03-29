#!/usr/bin/env python3)

# to begin, we will do sanity checks.
# at this point, we should have gdb loaded, this should be pulled
# in after, and we can process.  But let's be sure that we have 
# a valid inferior

import gdb
import os
import sys
print('python version is: ' + str(sys.version_info))
sys.path.append(os.getcwd())
from enum import Enum
from operator import itemgetter

import helpers
from helpers import divergencies, analyzed



def analyze_div(num, div):
    """This function performs an analysis on the 
    QC div record passed in -- that is, it transforms
    the QC record to a QD record (finding the point of
    divergence in the variable of interest's value)"""
    retVal = None
    sub = helpers.subjects
    #set command lines
    #start infs
    estate = helpers.execState.search1
    inf1 = gdb.selected_inferior().num
    watch1 = sub.getWatch(inf1)
    inf2 = sub.getOtherInf()
    watch2 = sub.getWatch(inf2)
    curInf = inf1
    curW = watch1
    #we're ready to search
    helpers.execCommands(['set logging file /dev/null', 'set logging redirect on', 'set logging on'])
    helpers.execCommands(['inferior 1'])
    print('inf1 is: ' + str(inf1) + ' and inf2 is: ' + str(inf2))
    print('watch1 inf is: ' + str(watch1.inf) + ' and watch2 inf is: ' +
          str(watch2.inf))
    while True:
        if estate == helpers.execState.search1:
            print('handling search1 state')
            print('inf1 masterCount is: ' + str(watch1.masterCount))
            helpers.execCommands(['continue'])
            if (watch1.state == helpers.watchState.hitCount or
                watch1.state == helpers.watchState.infExited):
                estate = helpers.execState.search2
                sub.toggle_inf()  #activates the other inferior
            # else:
            #     raise gdb.error('reached unknown state after helpers.execState.search1')
        if estate == helpers.execState.search2:
            print('handling search2 state')
            print('inf2 masterCount is: ' + str(watch2.masterCount))
            helpers.execCommands(['continue'])
            if (watch2.state == helpers.watchState.hitCount or
                watch2.state == helpers.watchState.infExited):
                estate = helpers.execState.analyze
            # else:
            #     raise gdb.error('reached unknown state after helpers.execState.search2')
        if estate == helpers.execState.analyze:
            print('handling analyze state')
            #TODO we need to get the first divergence here and construct
            #a new record (from QC data to analyzed)
            #need to get source file, source line, and memory address (instruction location)
            adiv = sub.seekDivergence()
            if adiv != None:
                print('divergence located: ' + str(adiv))
                #div.extend(adiv)
                retVal =  div + adiv
                break
            else:
                if (watch1.state == helpers.watchState.infExited  or
                    watch2.state == helpers.watchState.infExited):
                    raise gdb.error('A divergence was indicated by Classifier; none found @: ' +
                                    str(div))
                else:
                    sub.setSearching() #search the next block
    helpers.execCommands(['set logging off'])
    return retVal
                
def register_commands():
    helpers.InfoDivergenciesCommand()
    helpers.SeekDivergenceCommand()
    helpers.WriteQDDataCommand()
    helpers.LoadQDDataCommand()

def write_QD_file(filename):
    with open(filename) as f:
        f.write(repr(divergencies))

def init_inferiors():
    """This function creates the 2nd inferior,
    and sets the executable for each inferior.
    Precondition: inferior 2 doesn't yet exist"""
    helpers.execCommands(['file inf1',
                          'add-inferior -exec inf2'])

def analyze_all(div_info):
    for i, div in enumerate(div_info):
        helpers.prep_watches(div)
        divergencies.append(analyze_div(i, div))
        #if i == 10: break
        
def analyzed(div_info):
    if len(div_info) > 0:
        print('len of div_info[0] is: ' + str(len(div_info[0])))
        return not len(div_info[0]) == 5
    else:
        print('div_info is empty')


div_info = []

def main():
    print('hit main')
    register_commands()
    init_inferiors()
    
    gdb.events.exited.connect(helpers.catch_term)
    helpers.subjects = helpers.qfpSubject()
    global div_info
    if ("PARAMS" in os.environ):
        div_info = helpers.read_file(os.environ['PARAMS'])
        if analyzed(div_info): 
            divergencies.extend(div_info)
        else:
            analyze_all(div_info)
    else:
        print('No QD|QC data file was specified.  You may load QD with \'load [path]\'')
    #return to user prompt
    
if __name__ == "__main__":
    main()

  
