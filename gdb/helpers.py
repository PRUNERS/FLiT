#here are the helper functions / classes for geb_extends

import gdb
# import testEvents
# import pprint
import re
import _thread

#here are the data structures we need to record watchpoint hit data
#to determine where execution pairs diverge

infVals = [[],[]]
CPERIOD = 100


class qfpWatchpoint (gdb.Breakpoint):
    spec = None
    dtype = None
    count = 0
    target = -1
    replay = False
    def __init__(self, spec, dtype):
        super(qfpWatchpoint, self).__init__(spec, gdb.BP_WATCHPOINT,
                                            internal = False)
        spec = spec
        dtype = dtype
        
    def stop (self):
        if self.count == self.target:
            print('reached divergence point')
            return true
        else:
            if replay:
                return false
        #print('hit qfpWatchpoint')
        global infVals
        inf = gdb.selected_inferior().num
        val = gdb.parse_and_eval('*(' + dtype + ' *)' + spec)
        infVals[inf].append(val)
        print('recorded: ' + val)
        self.count = self.count + 1
        return true
        #return mismatched
        #here we hit and we have to decide whether or not to continue


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
    wplist.append(qfpWatchpoint('*(' + wtype + ' *) ' + addr, wtype))

    # gdb.execute('record full')

    if trapped1 and trapped2:
        gdb.events.stop.disconnect(catch_trap)

    # pp = pprint.PrettyPrinter()
    # pp.pprint(event)
    # testEvents.info(event)
    #print('caught: ' + event.stop_signal)
