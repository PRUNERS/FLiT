#here are the helper functions / classes for geb_extends

import gdb
# import testEvents
# import pprint
import re
import _thread

#here are the data structures we need to record watchpoint hit data
#to determine where execution pairs diverge

infVals = {}
CPERIOD = 100



class qfpWatchpoint (gdb.Breakpoint):
    spec = None
    dtype = None
    addr = None
    count = 0
    target = -1
    replay = False
    inf = 0
    totalCount = 0
    label = ''
    global infVals
    def __init__(self, addr, dtype, label):
        self.spec = '*(' + dtype + '*)' + addr
        super(qfpWatchpoint, self).__init__(self.spec, gdb.BP_WATCHPOINT,
                                            internal = False)
        self.dtype = dtype
        self.addr = addr
        self.inf = gdb.selected_inferior().num
        self.label = label
        if addr not in infVals:
            infVals[addr] = [[],[]]
        
    def stop (self):
        """
        self.count counts hits (reset at rerun).
        self.target is the execution we're trying
        to locate (after previously detecting a divergence
        at this point).  CPERIOD is the number of hits
        where we'll record the watched value. When
        count == CPERIOD, we'll compare
        the results.  If no diff is found,
        then it will continue execution of each inf.
        if a diff is located, then self.target for
        each qfpWatchpoint of this address will
        be set with the desired hit index, their count
        set to 0 and record will be enabled.  Then each inf
        will be restarted, and the stop() will
        return True when count == self.target,
        putting each inf at the point of divergence and
        in a gdb context that can be explored.
        """
        if self.count == self.target:
            print('reached divergence point')
            return True
        else:
            if self.replay:
                return False
        #print('hit qfpWatchpoint')
        print('handling inf: ' + str(self.inf))
        print(self.spec)
        val = gdb.parse_and_eval(self.spec)
        print('new val is: ' + str(val))
        infVals[self.addr].append(val)
        self.count = self.count + 1
        return True
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

wplist = {} #(one for each inferior per address)

def catch_trap(event):
    global wplist
    cur = gdb.selected_inferior()

    print('caught int3 in inf ' + str(cur.num))
    f = open('inf' + str(cur.num) + '.watch', 'r')
    wdata = f.read()
    print('read watch file: ' + wdata)
    m = re.match(r"[*]+checkAddr:(\w+)\n[*]+checkLen:(\w+)\n" +
                 "[*]+checkLab:(\w+)\n",
                 wdata)
    addr = m.group(1).strip()
    leng = m.group(2).strip()
    lab  = m.group(3).strip()
    print('addr: ' + addr + '; len: ' + leng + '; label: ' + lab)
    wtype = ''

    if leng == '8':
        wtype = 'double'
    else:
        wtype = 'float'

    if addr not in wplist:
        wplist[addr] = []    

    wplist[addr].append(qfpWatchpoint(addr, wtype, lab))
    assert len(wplist[addr] <= 2) #enforce 1 per inf per address

    print('set watchpoint @' + addr + ', type: ' + wtype + ', label: ' + lab)

    f.close()
    
    open('inf' + str(cur.num) + '.watch', 'w').close() #erase watch data

def inf_terminated(inf):
    infs = gdb.inferiors()
    if len(infs) < inf return False
    for t in gdb.inferiors()[inf - 1]:
        if t.is_valid() return False
    return True
    
def toggle_inf():
    to = (gdb.selected_inferior() + 1) % 1
    gdb.execute('inferior ' + to)

def 
    
