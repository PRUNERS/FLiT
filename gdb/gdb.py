#!/usr/bin/env python3

#this is the redesigned gdb extension for
#QD, quick differential debugger


#There are a few states and events
#that are handled in this FSM.

import gdb
import re
from enum import Enum
import sys
import inspect
from os import getcwd
# sys.path.append(getcwd())
#import helpers

DEBUG = False

class states(Enum):
    initial = 1
    start1 = 2
    start2 = 3
    run1 = 4
    run2 = 5
    analyze = 6
    seek = 7
    user = 8
    term = 9

class events(Enum):
    startup = 1
    i1Term = 2
    i2Term = 3
    i1Int3 = 4
    i2Int3 = 5
    i1Watch = 6
    i2Watch = 7
    empty = 8

# state variables
state = -1
watches = []
watchHits = [[],[]] #[[value, function],...]

#state functions
def initialS(e):
    debugMsg('hit ' + str(inspect.stack()[0][3]) + ' with event: ' + str(e))
    global state
    if e == events.startup:
        gdb.events.stop.connect(catchInt3)
        gdb.events.exited.connect(catchTerm)
        state = states.start1
        execCommands(['run 2> inf1.watch'])
    else:
        handleEventError(e, inspect.stack()[0][3])

def start1S(e):
    debugMsg('hit ' + str(inspect.stack()[0][3]) + ' with event: ' + str(e))
    global state
    if e == events.i1Int3:
        createWatch(1)
        state = states.start2
        execCommands(['add-inferior -exec inf2',
                              'inferior 2', 'run 2> inf2.watch'])
    elif e == events.i1Term:
        print('***** Inferior 1 terminated.  There was no interrupt ' +
              'for initializing watchpoint')
        state = states.term
    else:
        handleEventError(e, inspect.stack()[0][3])

def start2S(e):
    debugMsg('hit ' + str(inspect.stack()[0][3]) + ' with event: ' + str(e))
    global state
    if e == events.i2Int3:
        createWatch(2)
        gdb.events.stop.disconnect(catchInt3)
        state = states.run1
        execCommands(['inferior 1', 'continue'])
    elif e == events.i2Term:
        print('***** Inferior 2 terminated.  There was no interrupt ' +
              'for initializing watchpoint')
        state = states.term
    else:
        handleEventError(e, inspect.stack()[0][3])

def run1S(e):
    debugMsg('hit ' + str(inspect.stack()[0][3]) + ' with event: ' + str(e))
    global state
    if e == events.i1Watch:
        recordWatchHit(1)
        #state remains the same
    elif e == events.i1Term:
        watches[0].delete()
        state = states.run2
        execCommands(['inferior 2', 'continue'])
#        execCommands(['continue']) #I don't know why this is necessary, but the first continue is ignored
    else:
        handleEventError(e, inspect.stack()[0][3])
        
def run2S(e):
    debugMsg('hit ' + str(inspect.stack()[0][3]) + ' with event: ' + str(e))
    global state
    if e == events.i2Watch:
        recordWatchHit(2)
        #state remains the same
    elif e == events.i2Term:
        state = states.analyze
        handleEvent(events.empty)
    else:
        handleEventError(e, inspect.stack()[0][3])

def userS(e):
    debugMsg('hit ' + str(inspect.stack()[0][3]) + ' with event: ' + str(e))
    print('handining event ' + e + ' in qd in user state')

def analyzeS(e):
    debugMsg('hit ' + str(inspect.stack()[0][3]) + ' with event: ' + str(e))
    divs = []
    # for e in watchHits[0]:
    #     print(str(e))
    # for e in watchHits[1]:
    #     print(str(e))
    # print(str(watchHits))
    for i, elem in enumerate(zip(watchHits[0], watchHits[1])):
        #print(str(elem[0][0]) + ":"+ str(elem[0][0]))
        if elem[0][0] != elem[1][0]:
            divs.append(i)
            print(str(i) + ":")
            print(str(elem[0][0]) + ',' + elem[0][1] + ':')
            print(str(elem[1][0]) + ',' + elem[1][1])
        
def seekS(e):
    debugMsg('hit ' + str(inspect.stack()[0][3]) + ' with event: ' + str(e))
    raise gdb.error('seekS not implemented')
        
def termS(e):
    debugMsg('hit ' + str(inspect.stack()[0][3]) + ' with event: ' + str(e))
    gdb.events.stop.disconnect(catchInt3)
    gdb.events.exited.disconnect(catchTerm)
    print('handling event ' + e + 'in terminated state. ' +
          'Disconnected external event handlers')

dispatchS = {
    states.initial: initialS,
    states.start1: start1S,
    states.start2: start2S,
    states.run1: run1S,
    states.run2: run2S,
    states.analyze: analyzeS,
    states.seek: seekS,
    states.user: userS,
    states.term: termS
    }

#other helpers

def debugMsg(msg):
    if DEBUG:
        print(msg)

def recordWatchHit(inf):
    debugMsg('hit recordWatchHit with inf ' + str(inf))
    #print('watch val: ' + str(gdb.parse_and_eval(watches[inf-1].spec)))
    watchHits[inf-1].append([float(gdb.parse_and_eval(watches[inf-1].spec)),
                      gdb.newest_frame().name()])

def execCommands(clist):
    for c in clist:
        print('executing: ' + c)
        #gdb.flush()
        gdb.post_event(CommandEvent(c))

def handleEvent(e):
    dispatchS[state](e)

def handleEventError(e, f):
    raise gdb.error('*****unhandled event ' + str(e) + ' passed to ' + str(f))

def createWatch(inf):
    debugMsg('hit createWatch with inf ' + str(inf))
    wdata = ''
    addr = ''
    leng = ''
    lab = ''
    file = 'inf' + str(inf) + '.watch'
    f = None
    try:
        f = open(file, 'r')
        try:
            wdata = f.read()
        finally:
            f.close()
    except IOError:
        raise gdb.error('*****Error opening file: ' + file)
    match = re.match(r"[*]+checkAddr:(\w+)\n[*]+checkLen:(\w+)\n" +
                     "[*]+checkLab:(\w+)\n",
                     wdata)
    if match is not None:
        addr = match.group(1).strip()
        leng = match.group(2).strip()
        lab  = match.group(3).strip()
    else:
        raise gdb.error('*****Error matching re in watch file: ' + file)
    watches.append(qdWatchpoint(addr, lab, leng, inf))

def getTypeStr(s):
    if s == 4:
        return 'float'
    elif s == 8:
        return 'double'
    else:
        return 'long double'
    

# Here are external event handlers, int3 and term.  Watch
# hits are handled by qdWatchpoint class
def catchInt3(gEvent):
    debugMsg('hit catchInt3 with event: ' + str(gEvent))
    if type(gEvent) == gdb.BreakpointEvent:
        return
    else:
        if gdb.selected_inferior().num == 1:
            handleEvent(events.i1Int3)
        else:
            handleEvent(events.i2Int3)

def catchTerm(event):
    debugMsg('hit catchTerm with event: ' + str(event))
    if gdb.selected_inferior().num == 1:
        handleEvent(events.i1Term)
    else:
        handleEvent(events.i2Term)
        
class qdWatchpoint (gdb.Breakpoint):
    def __init__(self, addr, descr, leng, inf):
        print('in qdWatchpoint init, leng is: ' + leng + ' getTypeStr(leng) is: ' + getTypeStr(int(leng)))
        self.spec = '*(' + getTypeStr(int(leng)) + '*)' + addr
        self.descr = descr
        self.inf = inf        
        super(qdWatchpoint, self).__init__(self.spec, gdb.BP_WATCHPOINT, gdb.WP_WRITE,
                                           internal = False)
    def stop (self):
        debugMsg('hit qdWatchpoint:stop in inf: ' + str(self.inf))
        if self.inf == 1:
            handleEvent(events.i1Watch)
        else:
            handleEvent(events.i2Watch)
        return False
    
    def delete (self):
        gdb.Breakpoint.delete(self)

    def reinit (self):
        super(qdWatchpoint, self).__init__(self.spec, gdb.BP_WATCHPOINT, gdb.WP_WRITE,
                                           internal = False)

class CommandEvent():
    def __init__(self, cmd):
        self.cmd = cmd
    def __call__(self):
        #try:
        gdb.execute(self.cmd)
        #I'm not sure we want to catch this.  Perhaps it will be ignored /
        #handled silently otherwise . . .
        # except gdb.error as e:
        #     print('caught gdb.error: ' + str(e) + ' on gdb.execute(' +
        #           c + ')')

def main():
    debugMsg('cwd is: ' + getcwd())
    global state
    state = states.initial 
    handleEvent(events.startup)

if __name__ == "__main__":
    main()
