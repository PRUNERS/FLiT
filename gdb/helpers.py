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
    infExited = 7
    
subjects = {}

class qfpWatchpoint (gdb.Breakpoint):
    spec = None
    dtype = None
    addr = None
    count = 0
    target = -1
    inf = 0
    masterCount = 0
    state = watchState.searching
    values = []
    funcs = []
    subject = None
    global infVals
    def __init__(self, addr, dtype, label, subject):
        self.spec = '*(' + dtype + '*)' + addr
        super(qfpWatchpoint, self).__init__(self.spec, gdb.BP_WATCHPOINT,
                                            internal = False)
        self.dtype = dtype
        self.addr = addr
        self.inf = gdb.selected_inferior().num
        self.subject = subject

    def setSearching(self):
        self.masterCount += self.count
        self.count = 0
        self.values = []
        self.target = -1

    def setSeeking(self, target):
        self.state = watchState.seeking
        self.masterCount += self.count
        self.count = 0
        self.target = target

    def setHitSeek(self):
        print('reached divergence point')
        self.state = watchState.hitSeek

    def setHitCount(self):
        self.state = watchState.hitCount
        self.masterCount += self.count
        self.count = 0

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
            if self.state == watchState.seeking:
                return False
        print('handling inf: ' + str(self.inf) + ' at count: ' + str(self.count))
        print(self.spec)
        val = gdb.parse_and_eval(self.spec)
        print('new val is: ' + str(val))
        self.values.append(val)
        self.funcs.append(gdb.newest_frame().name())
        self.count = self.count + 1
        if self.count == self.subject.CPERIOD:
            self.setHitCount()
            return True
        else:
            return False
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
        self.watches.append(qfpWatchpoint(addr, wtype, label, self))

    def getDivergence(self):
        vals = []
        funs = []
        divs = []
        for cnt, vals in enumerate(zip(self.watches[0].values, self.watches[1].values, self.watches[0].funcs, self.watches[1].funcs)):
            print('comparing v1 v2, f1, f2:' + str(vals[0]) + ':' + str(vals[1]) +
                  ':' + str(vals[2]) + ':' + str(vals[3]))
            if vals[0] != vals[1] or vals[2] != vals[3]:
                vals.append([vals[0], vals[1]])
                funs.append([vals[2], vals[3]])
                divs.append(cnt)
#                    return cnt, vals[0].value, vals[1].value, vals[0].fun, vals[1].fun
        return vals, funs, divs

    def seekDivergence(self):
        vals, funs, divs = self.getDivergence()
        print('in seekDivergence, vals is: ' + str(vals))
        if len(vals) > 0:
            for v,f,d in zip(vals, funs, divs):
                # choice = input('variable divergence in subject ' + sub.label +
                #                '; iteration: ' + d + '; functions: ' +
                #                f[0] + ':' + f[1] + '; values: ' +
                #                v[0] + ':' + v[1] + '.  _F_ocus or _I_gnore?')
                # if choice.lower() != 'f' and choice.lower() != 'focus':
                    return d
        return -1
        
    # here we should update watches
    # clear values, advance masterCount
    # and zero count
    # also set the state
    def setSearching(self):
        self.state = subjState.searching
        for w in self.watches:
            w.setSearching()
                
    # set record, restart inferiors,
    # reset counts and targets
    def setSeeking(self, target):
        self.state = subjState.seeking
        for w in self.watches:
            w.setSeeking(target)

    def getWatch(self, inf):
        for w in self.watches:
            if w.inf == inf:
                return w
        gdb.error('inf ' + inf + ' not found in getWatch()')

    def getOtherInf(self):
        cur = gdb.selected_inferior().num
        for w in self.watches:
            if w.inf != cur:
                return w.inf
        gdb.error('couldn\'t locate other inf in subject: ' +
                  sub.label)
        
    def toggle_inf(self):
        execCommands(['inferior ' + str(self.getOtherInf())])
    
def execCommands(clist):
    for c in clist:
        print('executing: ' + c)
        gdb.flush()
        try:
            gdb.execute(c)
        except gdb.error as e:
            print('caught gdb.error: ' + str(e) + ' on gdb.execute(' +
                  c + ')')

def getPrecString(p):
    if p == 'f':
        return 'float'
    if p == 'd':
        return 'double'
    return 'long double'

#wplist = {} #(one for each inferior per address)



def catch_trap(event):
    global subjects
    cur = gdb.selected_inferior()
    if type(event) == gdb.BreakpointEvent:
        return
    print('stop caught: ' + str(event))
    print('event type: ' + str(type(event)))
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
    print('hit next print')

    if lab not in subjects:
        subjects[lab] = qfpSubject(lab, cur.num, addr, wtype)
    else:
        subjects[lab].watches.append(qfpWatchpoint(addr, wtype, lab,
                                              subjects[lab]))

    print('subjects length is: ' + str(len(subjects)))

    if len(subjects[lab].watches) == 2:
       subjects[lab].setSearching()
       #TODO need to handle this better
       gdb.events.stop.disconnect(catch_trap)

    print('set watchpoint @' + addr + ', type: ' + wtype + ', label: ' + lab)

    f.close()
    
    open('inf' + str(cur.num) + '.watch', 'w').close() #erase watch data

def inf_terminated(inf):
    infs = gdb.inferiors()
    if len(infs) < inf: return False
    for t in gdb.inferiors()[inf - 1]:
        if t.is_valid(): return False
    return True

def catch_term(event):
    global subjects
    inf = event.inferior.num
    print('caught term signal in inf ' + str(inf))
    for k, s in subjects.items():
        for w in s.watches:
            print('checking if ' + str(w.inf) + ' matches ' + str(inf))
            if w.inf == inf:
                w.state = watchState.infExited
            
