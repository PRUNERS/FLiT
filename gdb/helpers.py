#here are the helper functions / classes for geb_extends

import gdb
import re
import copy
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

def copy_qfpWatchpoint(orig):
    print('hit copy_qfpWatchpoint')
    return qfpWatchpoint(orig.addr, orig.dtype,
                         orig.subject,
                         orig.spec, orig.count,
                         orig.target, orig.inf,
                         orig.masterCount, orig.state,
                         orig.values, orig.funcs)

def infBeyondMain():
    # we need to walk the stack to the bottom to make sure we
    # haven't returned from main.  May be expensive
    curFrame = gdb.newest_frame()
    print('in infBeyondMain, curFrame is: ' + str(curFrame) + ', curFrame.name:' + curFrame.name())
    print('\t and curFrame.function() is: ' + str(curFrame.function()))
#    while curFrame != None and curFrame.function() != None:
    while curFrame != None and curFrame.name() != None:
        print('in infBeyondMain, curFrame: ' + str(curFrame), ' name: ' + str(curFrame.name()))
#        print('infBeyondMain(): frame is ' + curFrame.function().name)
# This is a total hack for NCAR KGen.  ifort doesn't have a frame named main, so we'll
# cheat and use the top level function in a KGen kernel: kernel_driver
        if (curFrame.name() == 'main' or
            curFrame.name() == 'main(int, char**)' or
            curFrame.name() == 'kernel_driver'):
        # if (curFrame.function().name == 'main' or
        #     curFrame.function().name == 'main(int, char**)' or
        #     curFrame.function().name == 'kernel_driver'):
            return False
        else:
            curFrame = curFrame.older()
    return True
    
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
    deleted = False
    def __init__(self, addr, dtype, subject, spec=None,
                 count=0, target=-1, inf=-1, masterCount=0,
                 state=watchState.searching, values=[], funcs=[]):
        print('hit qfpWatchpointr.__init__')
        if spec == None:
            self.spec = '*(' + dtype + '*)' + addr
        else:
            self.spec = spec

        if inf == -1:
            self.inf = gdb.selected_inferior().num
        else:
            self.inf = inf
        
        self.dtype = dtype
        self.addr = addr
        self.subject = subject
        self.count = count
        self.target = target
        self.masterCount = masterCount
        self.state = state
        self.values = copy.copy(values)
        self.funcs = copy.copy(funcs)
            
        super(qfpWatchpoint, self).__init__(self.spec, gdb.BP_WATCHPOINT,
                                            internal = False)

    def setSearching(self):
        print('in setSearching, count: ' + str(self.count) +
              ', masterCount: ' + str(self.masterCount))
        self.masterCount += self.count
        self.count = 0
        self.values = []
        self.target = -1

    def setSeeking(self, target):
        #self.state = watchState.seeking
        #self.masterCount += self.count
        self.count = 0
        self.target = target
        self.masterCount = 0

    def setHitSeek(self):
        print('reached divergence point in inf ' + str(self.inf))
        self.state = watchState.hitSeek

    def setHitCount(self):
        self.state = watchState.hitCount
        # self.masterCount += self.count
        #self.count = 0

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
        print('hit qfpWatch.stop')
        print('count is ' + str(self.count) + ', masterCount is ' + str(self.masterCount))
        print('target is ' + str(self.target))
        # We need to ignore changes made outside of main (i.e. after main exits)
        if infBeyondMain():
            print('detected that main has returned in qfpWatch.stop()')
            return True
        if self.count == self.target:
            self.setHitSeek()
            return True
            # if self.subject.allAtSeek():
            #     return True
            # else:
            #     return False
        else:
            if self.state == watchState.seeking:
                self.count += 1
                return False
        print('handling inf: ' + str(self.inf) + ' at count: ' +
              str(self.count + self.masterCount))
        print('in func: ' + str(gdb.newest_frame().name))
        print(self.spec)
        val = gdb.parse_and_eval(self.spec)
        print('new val is: ' + str(val))
        self.values.append(val)
        self.funcs.append(gdb.newest_frame().name)
        self.count += 1
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
        self.watches.append(qfpWatchpoint(addr, wtype, self))
        
    def replaceWatch(self, inf):
        print('hit replaceWatch')
        for c, w in enumerate(self.watches):
            print('replaceWatch, watch: ' + str(w))
            if w.inf == inf:
                tmp = copy_qfpWatchpoint(w)
                if not w.deleted:
                    print('not deleted')
 #                   w.delete()
#                self.watches[c] = copy_qfpWatchpoint(w)
                self.watches[c] = tmp
                print('copied watch')
                self.watches[c].state = watchState.seeking

    def printValues(self):
        print('printValues:')
        for v1, v2 in zip(self.watches[0].values, self.watches[1].values):
            print(str(v1) + ':' + str(v2))
                
    def getDivergence(self):
        values = []
        funs = []
        divs = []
        print('in getDivergence, len(watches[0].values) is: '

              + str(len(self.watches[0].values)))
        self.printValues()
        print('\tlen(watches[1].values) is: ' + str(len(self.watches[1].values)))
        for cnt, vals in enumerate(zip(self.watches[0].values, self.watches[1].values, self.watches[0].funcs, self.watches[1].funcs)):
            #print('comparing v1 v2, f1, f2:' + str(vals[0]) + ':' + str(vals[1]) +
                  #':' + str(vals[2]) + ':' + str(vals[3]))
            # print('cnt + self.watches[0].masterCount: ' +
            #       str(cnt + self.watches[0].masterCount - self.CPERIOD))
            if (vals[0] != vals[1]):
            # or
            #     ((vals[2].find(vals[3])) == -1 and
            #     (vals[3].find(vals[2])) == -1)):
                # print('vals[0]:vals[1], ' +
                #       '2.find3:3.find2; ' +
                #       str(vals[0]) + ":" + str(vals[1]) + "; " +
                #       str(vals[2].find(vals[3])) + ":" +
                #       str(vals[3].find(vals[2])))
                
                values.append([vals[0], vals[1]])
                funs.append([vals[2], vals[3]])
                divs.append(cnt + self.watches[0].masterCount)
                break
#                    return cnt, vals[0].value, vals[1].value, vals[0].fun, vals[1].fun
        return values, funs, divs

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
    def setSeeking(self, target, cmdLine1, cmdLine2):
        self.state = subjState.seeking
        #we need to rest things -- delete watch points and let
        #catch_trap add the new watchpoints that are
        #copied from the old ones
        print('connecting catch_trap in setSeeking')
        gdb.events.stop.connect(catch_trap)
        print('set catch_trap')
        self.toggle_inf()
        print('running 1 to establish watch')
        #TODO we removed record, I suspect at switching inf for next seek,
        #we'll have to save the state, stop, and rerecord for inf2
        #(there is a problem with threads disappearing in inf2)
#        execCommands(['run 2> inf' + str(self.watches[0].inf) + '.watch'])
        execCommands(['run ' + cmdLine1])
        print('finished 1 for watch')
        self.toggle_inf()
        print('running 2 to establish watch')
#        execCommands(['run 2> inf' + str(self.watches[1].inf) + '.watch'])
        execCommands(['run ' + cmdLine2])
        print('finished 2 to watch')
        self.toggle_inf()
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
                  self.label)
        
    def toggle_inf(self):
        execCommands(['inferior ' + str(self.getOtherInf())])
        
    def allAtSeek(self):
        print('in allAtSeek, w1 state, w2 state: ' + str(self.watches[0].state),
              str(self.watches[1].state))
        for w in self.watches:
            if w.state != watchState.hitSeek:
                return False
        return True
    
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
    print('entered catch_trap')
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
    m = re.match(r"[*\s]+checkAddr:[\s]*(\w+)\n[*\s]+checkLen:[\s]*(\w+)\n" +
                 "[*\s]+checkLab:[\s]*(\w+)\n",
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
        if subjects[lab].state == subjState.seeking:
            subjects[lab].replaceWatch(cur.num)
        else:
            subjects[lab].watches.append(qfpWatchpoint(addr, wtype, 
                                              subjects[lab]))

    print('subjects length is: ' + str(len(subjects)))

    if (len(subjects[lab].watches) == 2 and
        (not (subjects[lab].watches[0].state == watchState.seeking) !=
         (subjects[lab].watches[1].state == watchState.seeking))):
        if subjects[lab].state == subjState.loading:
            subjects[lab].setSearching()
        gdb.events.stop.disconnect(catch_trap)

    print('set watchpoint @' + addr + ', type: ' + wtype + ', label: ' + lab)

    f.close()
    
    open('inf' + str(cur.num) + '.watch', 'w').close() #erase watch data

def inf_terminated(inf):
    print('entered inf_terminated')
    infs = gdb.inferiors()
    if len(infs) < inf: return False
    for t in gdb.inferiors()[inf - 1]:
        if t.is_valid(): return False
    return True

def catch_term(event):
    print('entered catch_term')
    global subjects
    inf = event.inferior.num
    print('caught term signal in inf ' + str(inf))
    for k, s in subjects.items():
        for w in s.watches:
            print('checking if ' + str(w.inf) + ' matches ' + str(inf))
            if w.inf == inf:
                w.state = watchState.infExited
                w.delete()
                return
                #this may be bad, but trying to delete the watchpoint but keep
                #our (subclass) data.  Docs say this deletes the super : P
            
