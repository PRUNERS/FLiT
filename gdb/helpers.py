#here are the helper functions / classes for geb_extends

import gdb
# import testEvents
# import pprint
import re
#import _thread
from enum import Enum

#here are the data structures we need to record watchpoint hit data
#to determine where execution pairs diverge


class subjState(Enum):
    loading = 1
    searching = 2
    seeking = 3
    hitEDiv = 3
    hitVDiv = 4

class watchState(Enum):
    searching = 1
    seeking = 2
    hitSeek = 5
    hitCount = 6
    
subjects = {}

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
    state = watchState.searching
    values = []
    global infVals
    def __init__(self, addr, dtype, label):
        self.spec = '*(' + dtype + '*)' + addr
        super(qfpWatchpoint, self).__init__(self.spec, gdb.BP_WATCHPOINT,
                                            internal = False)
        self.dtype = dtype
        self.addr = addr
        self.inf = gdb.selected_inferior().num
        self.label = label

    def getState(self):
        return state

    def setSeeking(self, target):
        self.state = watchState.seeking
        self.masterCount += w.count
        self.count = 0
        self.target = target

    def setHitSeek(self):
        print('reached divergence point')
        self.state = watchState.hitSeek

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
            self.setHitSeek()
            return True
        else:
            if self.replay:
                return False
        #print('hit qfpWatchpoint')
        print('handling inf: ' + str(self.inf))
        print(self.spec)
        val = gdb.parse_and_eval(self.spec)
        print('new val is: ' + str(val))
        self.values.append(val)
        self.count = self.count + 1
        return True
        #return mismatched
        #here we hit and we have to decide whether or not to continue

class qfpSubject:
    label = ''
    state = subjState.loading
    CPERIOD = 100
    watches = []
    def __init__(self, label, inf, addr, wtype):
        self.label =  label
        self.state = subjState.loading
        watches.append(qfpWatchpoint(addr, wtype, label))

    def getDivergence(self):
        for cnt, vals in enumerate(zip(watches[0]values, watches[1].values)):
            if vals[0].value != vals[1].value:
                if vals[0].fun != vals[1].fun:
                    
                return cnt, vals[0].value, vals[1].value, vals[0].fun, vals[1].fun
        return -1

    def seekDivergence(self):
        div, val0, val1, fun0, fun1 = sub.getDivergence()
        if div > -1:
            #TODO this is only handling the first divergence detected.  We should probably return a list
            choice = input('variable divergence in subject ' + sub.label +
                           '; iteration: ' + div + '; functions: ' +
                           fun0 + ':' + fun1 + '; values: ' +
                           val0 + ':' + val1 + '.  _F_ocus or _I_gnore?')
            return choice.lower() != 'f' and choice.lower() != 'focus'

        
    # here we should update watches
    # clear values, advance masterCount
    # and zero count
    # also set the state
    def setSearching(self):
        state = subjState.searching
        for w in watches:
            w.setSearching()
            w.masterCount += w.count
            w.count = 0
            w.values = []
            w.target = -1
                
    # set record, restart inferiors,
    # reset counts and targets
    def setSeeking(self, target):
        state = subjState.seeking
        for w in watches:
            w.setSeeking(target)
        commands = ['record', 'infer 1', 'run', 'infer 2', 'run']
        execCommands(commands)

    def getWatch(self, inf):
        for w in self.watches:
            if w.inf == inf:
                return w
        gdb.error('inf ' + inf ' not found in getWatch()')

    def getOtherInf():
        cur = gdb.selected_inferior()
        for w in self.watches:
            if w.inf != cur:
                return w.inf
        gdb.error('couldn\'t locate other inf in subject: ' +
                  sub.label)
        
    def toggle_inf(self):
        gdb.execute('inferior ' + self.getOtherInf())
    
    gdb.execute('inferior ' + to)
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

#wplist = {} #(one for each inferior per address)



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

    if lab in subjects:
        subjects[lab].setSearching()

    subjects[lab].watches.append(qfpSubject(lab, cur.num, addr, wtype))

    print('set watchpoint @' + addr + ', type: ' + wtype + ', label: ' + lab)

    f.close()
    
    open('inf' + str(cur.num) + '.watch', 'w').close() #erase watch data

def inf_terminated(inf):
    infs = gdb.inferiors()
    if len(infs) < inf return False
    for t in gdb.inferiors()[inf - 1]:
        if t.is_valid() return False
    return True
    

    
