#here are the helper functions / classes for geb_extends

import gdb
# import testEvents
# import pprint
import re

#here are the data structures we need to record watchpoint hit data
#to determine where execution pairs diverge

infVals = [[]]
count = [0,0]
CPERIOD = 100

class qfpWatchpoint (gdb.Breakpoint):
    spec = None
    dtype = None
    def __init__(self, spec, dtype):
        super(qfpWatchpoint, self).__init__(spec, gdb.BP_WATCHPOINT,
                                            internal = False)
        spec = spec
        dtype = dtype
        
    def stop (self):
        #print('hit qfpWatchpoint')
        global infVals, count
        inf = gdb.selected_inferior().num
        if count[inf + 1 % 1]
        count = count + 1
        return mismatched
        #here we hit and we have to decide whether or not to continue

def watch_handler (event):
    pass
    #if 

def execCommands(clist):
    for c in clist:
        print('executing: ' + c)
        gdb.execute(c)

def getPrecString(p):
    if p == 'f':
        return 'float'
    if p == 'd':
        return 'double'
    return 'long double'

wplist = []
trapped1 = False
trapped2 = False

def catch_trap(event):
    global trapped1, trapped2, wplist
    cur = gdb.selected_inferior()

    if cur.num == 1:
        if trapped1 == True:
            return
        trapped1 = True
    else:
        if trapped2 == True:
            return
        trapped2 = True
        

    print('caught int3 in inf ' + str(cur.num))
    f = open('inf' + str(cur.num) + '.watch', 'r')
    wdata = f.read()
    print('read watch file: ' + wdata)
    m = re.match(r"[*]+checkAddr:(\w+)\n[*]+checkLen:(\w+)\n",
                 wdata)
    addr = m.group(1)
    leng = m.group(2)
    print('addr: ' + addr + '; len: ' + leng)
    wtype = ''
    if leng == '8':
        wtype = 'double'
    else:
        wtype = 'float'
    wplist.append(qfpWatchpoint('*(' + wtype + ' *) ' + addr))

    gdb.execute('record full')

    if trapped1 and trapped2:
        gdb.events.stop.disconnect(catch_trap)

    # pp = pprint.PrettyPrinter()
    # pp.pprint(event)
    # testEvents.info(event)
    #print('caught: ' + event.stop_signal)
