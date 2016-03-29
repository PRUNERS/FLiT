#here are the helper functions / classes for geb_extends

import gdb
import re
import copy
from enum import Enum
from os import path
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
    
subjects = None

divergencies = []
analyzed = []

def prep_watches(div):
    """This function sets up the inferiors, and hands them off to the 
    handlers for int3, thereby setting up the watchpoints"""

    for w in subjects.watches:
        if w.is_valid():
            w.delete()
    subjects.watches = []
    

    execCommands(['inferior 1'])
        #this is the watchpoint setup routine
    gdb.events.stop.connect(catch_trap)
    #this is disconnected when the watch is set
    print('in prep_watches, div is: ' + str(div))
    execCommands(['run ' + div[1] + ' ' + str(div[2]) + ' ' +
                          str(div[3]) + ' ' + div[0] + ' 2> inf1.watch'])

    execCommands(['inferior 2'])
    #this is the watchpoint setup routine
    gdb.events.stop.connect(catch_trap)
    #this is disconnected when the watch is set

    execCommands(['run ' + div[1] + ' ' + str(div[2]) + ' ' +
                          str(div[3]) + ' ' + div[0] + ' 2> inf2.watch'])
    execCommands(['infer 1'])
    
def copy_qfpWatchpoint(orig):
    print('hit copy_qfpWatchpoint')
    return qfpWatchpoint(orig.addr, orig.dtype,
                         orig.subject,
                         orig.spec, orig.count,
                         orig.target, orig.inf,
                         orig.masterCount, orig.state,
                         orig.values, orig.funcs)

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
        print('in setSeeking, target is: ' + str(target))
        self.count = 0
        self.target = target
        self.masterCount = 0
        self.state = watchState.seeking

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
        if infBeyondMain():
            print('detected that main has returned in qfpWatch.stop()')
            return True
        if self.count == self.target:
            self.setHitSeek()
            return True
        else:
            if self.state == watchState.seeking:
                self.count += 1
                return False
        print('before handling inf')
        print('handling inf: ' + str(self.inf) + ' at count: ' +
              str(self.count + self.masterCount))
        print('in func: ' + str(gdb.newest_frame().name))
        print(self.spec)
        val = gdb.parse_and_eval(self.spec)
        print('new val is: ' + str(val))
        self.values.append(val)
        gdb.post_event(CurrentLocation(self.count, self.values))
        self.count += 1
        if self.count == self.target:
            self.setHitSeek()
        return True

def infBeyondMain():
    # we need to walk the stack to the bottom to make sure we
    # haven't returned from main.  May be expensive
    #TODO for now we're skipping this check -- it should be ok if
    #we're only looking for the FIRST divergence
    return False
    curFrame = gdb.newest_frame()
    print('hi from infbeyondmain')
    print('curFrame is: ' + str(curFrame))
    #delme
    #raise gdb.error('we\'re stopping in infBeyondMain')
    if curFrame == None:
        print('curFrame is None')

    print('cf.name() is: ' + str(curFrame.name()))
    print('in infBeyondMain, curFrame is: ' + str(curFrame) + ', curFrame.name:' + str(curFrame.name()))
    print('\t and curFrame.function() is: ' + str(curFrame.function()))
    while curFrame != None and curFrame.name() != None:
        print('in infBeyondMain, curFrame: ' + str(curFrame), ' name: ' + str(curFrame.name()))
        if (curFrame.name() == 'main' or
            curFrame.name() == 'main(int, char**)' or
            curFrame.name() == 'kernel_driver'):
            return False
        else:
            curFrame = curFrame.older()
    return True

class qfpSubject:
    #label = ''
    state = subjState.loading
    CPERIOD = 100
    watches = []
    def __init__(self):
        self.watches = []
        # self.label =  label
        # self.state = subjState.loading
        #self.watches.append(qfpWatchpoint(addr, wtype, self))
        
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
                
    # def getDivergence(self):
    #     values = []
    #     print('in getDivergence, len(watches[0].values) is: '
    #           + str(len(self.watches[0].values)))
    #     self.printValues()
    #     print('\tlen(watches[1].values) is: ' + str(len(self.watches[1].values)))
    #     for cnt, vals in enumerate(zip(self.watches[0].values, self.watches[1].values)):
    #         if (vals[0] != vals[1]):
    #             values.append([vals[0], vals[1]])
    #             divs.append(cnt + self.watches[0].masterCount)
    #             break
    #     return values, divs

    def seekDivergence(self):
        for i, div in enumerate(zip(self.watches[0].values, self.watches[1].values)):
            if div[0][1] != div[1][1]:
                return [div[0],div[1]]
        return None
        
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
        #we need to rest things -- delete watch points and let
        #catch_trap add the new watchpoints that are
        #copied from the old ones
        prep_watches(target)
#         print('connecting catch_trap in setSeeking')
#         gdb.events.stop.connect(catch_trap)
#         print('set catch_trap')
#         self.toggle_inf()
#         print('running 1 to establish watch')
#         #TODO we removed record, I suspect at switching inf for next seek,
#         #we'll have to save the state, stop, and rerecord for inf2
#         #(there is a problem with threads disappearing in inf2)
# #        execCommands(['run 2> inf' + str(self.watches[0].inf) + '.watch'])
#         execCommands(['run ' + cmdLine1])
#         print('finished 1 for watch')
#         self.toggle_inf()
#         print('running 2 to establish watch')
# #        execCommands(['run 2> inf' + str(self.watches[1].inf) + '.watch'])
#         execCommands(['run ' + cmdLine2])
#         print('finished 2 to watch')
#         self.toggle_inf()
        print('executed prep watches')
        for w in self.watches:
            w.setSeeking(target[5][0])

    def getWatch(self, inf):
        for w in self.watches:
            if w.inf == inf:
                return w
        raise gdb.error('inf ' + inf + ' not found in getWatch()')

    def getOtherInf(self):
        cur = gdb.selected_inferior().num
        for w in self.watches:
            if w.inf != cur:
                return w.inf
        raise gdb.error('couldn\'t locate other inf in subject: ' +
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
    """This handler sets up watchpoints.  
    A precondition is that inf1 must be set before
    inf2.
    """
    print('entered catch_trap')
    global subjects
    cur = gdb.selected_inferior()
    if type(event) == gdb.BreakpointEvent:
        return
    print('stop caught: ' + str(event))
    print('event type: ' + str(type(event)))
    print('caught int3 in inf ' + str(cur.num))
    with open('inf' + str(cur.num) + '.watch', 'r') as f:
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

    # if subjects == None:
    #     subjects = qfpSubject()
    subjects.watches.append(qfpWatchpoint(addr, wtype, 
                                          subjects))

    gdb.events.stop.disconnect(catch_trap)
    print('set watchpoint @' + addr + ', type: ' + wtype + ', label: ' + lab)

    with open('inf' + str(cur.num) + '.watch', 'w') as f:
        f.close() #erase watch data

def inf_terminated(inf):
    print('entered inf_terminated')
    infs = gdb.inferiors()
    if len(infs) < inf: return False
    for t in gdb.inferiors()[inf - 1]:
        return not t.is_valid()
    return True

def catch_term(event):
    print('entered catch_term')
    global subjects
    inf = event.inferior.num
    print('caught term signal in inf ' + str(inf))
    subjects.watches[inf-1].state = watchState.infExited
    subjects.watches[inf-1].delete()
    #for k, s in subjects.items():
    # for w in subjects.watches:
    #     print('checking if ' + str(w.inf) + ' matches ' + str(inf))
    #     if w.inf == inf:
    #         w.state = watchState.infExited
    #         w.delete()
    #         return
                #this may be bad, but trying to delete the watchpoint but keep
                #our (subclass) data.  Docs say this deletes the super : P

def getDivHeader():
    return "id\tifile\tvar\ti1\ti2\terror\tsource File\tsource line\tiaddress"

def getDivStr(num, div): #used for the command
    #this is the layout:
    # num e1 e2 ifile var i1 i2 v1 v2 sfile sline iaddr
    retVal = (str(num) + '\t' +
              path.basename(div[0]) + '\t' +
              str(div[1]) + '\t' +
              str(div[2]) + '\t' +
              str(div[3]) + '\t' +
              str(div[4]) + '\t' +
              str(div[5][2]) + '\t' +
              str(div[5][4]) + '\t' +
              str(format(div[5][3], '#08x')))
    return retVal

def seekDiv(num):
    print('handling seek1 state')
    gdb.events.exited.disconnect(catch_term)
    subjects.setSeeking(divergencies[int(num)])
    gdb.events.exited.connect(catch_term)
    print('executed subjects setSeeking')
    watch1 = subjects.watches[0]
    watch2 = subjects.watches[1]
    inf1 = watch1.inf
    inf2 = watch2.inf
    estate = execState.seek1
    # watch1.state = watchState.seeking
    # watch2.state = watchState.seeking
    # watch1.target = 
    execCommands(['continue'])
                #DELME
    print('after continue, watch1.state and watch2.state are: ' +
        str(watch1.state) + ' and ' + str(watch2.state))
    while True:
        if watch1.state == watchState.hitSeek:
            estate = execState.seek2
            subjects.toggle_inf()
        # else:
        #     raise gdb.error('couldn\'t reach target in seek of inf ' + str(inf1))
        if estate == execState.seek2:
            print('handling seek2 state')
            #break
            execCommands(['continue'])
            if watch2.state == watchState.hitSeek:
                estate = execState.user
                break
            # else:
            #     raise gdb.error('couldn\'t reach target in seek of inf ' + str(inf2))

def read_file(filename):
    with open(filename) as f:
        return eval(f.read().replace('\n', '').replace(' ', ''))

    
            
# Here are our command definitions (for navigating divergencies)

class InfoDivergenciesCommand (gdb.Command):
    """List information on divergence locations"""

    def __init__ (self):
        super(InfoDivergenciesCommand, self).__init__ ("info divergencies",
                                                       gdb.COMMAND_SUPPORT,
                                                       gdb.COMPLETE_NONE)

    def invoke(self, arg, from_tty):
        ifile = None
        varName = None
        args = gdb.string_to_argv(arg)
        if len(args) == 2:
            ifile = args[1]
        if len(args) > 0:
            varName = args[0]
        print(getDivHeader())
        for i, div in enumerate(divergencies):
            if ((ifile != None and ifile in div[1]) or
                ifile == None):
              if ((varName != None and varName in div[2]) or
                varName == None):
                print(getDivStr(i, div))
        
class SeekDivergenceCommand(gdb.Command):
    """Go to point of divergence in full gdb context"""

    def __init__ (self):
        super(SeekDivergenceCommand, self).__init__ ("seek",
                                                     gdb.COMMAND_SUPPORT,
                                                     gdb.COMPLETE_NONE)
    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        if len(args) == 1:
            seekDiv(args[0])
        else:
            raise gdb.GdbError("seek takes one argument: divergence number.")

class WriteQDDataCommand(gdb.Command):
    """writes the current divergencies data to file"""
    def __init__(self):
        super(WriteQDDataCommand, self).__init__("write",
                                                     gdb.COMMAND_SUPPORT,
                                                     gdb.COMPLETE_NONE)

    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        if len(args) == 1:
            with open(args[0], 'w') as f:
                f.write(repr(divergencies))
        else:
            raise gdb.GdbError("You must supply a filename to write")

class LoadQDDataCommand(gdb.Command):
    """Loads a file with QD data (analyzed divergency list)"""
    def __init__(self):
        super(LoadQDDataCommand, self).__init__("load",
                                                     gdb.COMMAND_SUPPORT,
                                                     gdb.COMPLETE_NONE)

    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        global divergencies
        if len(args) == 1:
            divergencies = read_file(args[0])
        else:
            raise gdb.GdbError("You must supply a filename to load")

    
        
class CurrentLocation():
    """This is the event that allows us to get accurate location information
    when one of our watchpoints is hit.  If we try to access this from within
    Breakpoint.stop(), we get very inaccurate information.  We have to
    add this to the gdb event queue"""
    def __init__(self, hit_count, values):
        gdb.write('inside CurrentLocation.__init__\n')
        self.count = hit_count
        self.values = values
    def __call__(self):
        print('our event was taken from the queue!\n')
        print(gdb.decode_line()[1][0])
        print('wp count is: ' + str(self.count))
        file = gdb.decode_line()[1][0].symtab.filename
        line = gdb.decode_line()[1][0].line
        iaddr = gdb.decode_line()[1][0].symtab.linetable().line(line)[0].pc
        # print('file, line and iaddr are: ' + str(file) + ':' + str(line) + ':'
        #       + str(iaddr))
        # print('values len is: ' + str(len(self.values)))
        # print('values is: ' + str(self.values))
        #self.values[self.count].fetch_lazy()
        self.values[self.count] = [self.count, float(self.values[self.count]), file, iaddr, line]
        # print('after assignment, values is: ' + str(self.values))
        # print('just added: ' + self.values[self.count])
        # if self.watchpoint.count == self.watchpoint.subject.CPERIOD:
        #     self.watchpoinst.setHitCount()

#here are our exec states
class execState(Enum):
    init = 1
    search1 = 2
    search2 = 3
    analyze = 4
    seek1 = 5
    seek2 = 6
    user = 7

#sub = subjects[0]
estate = execState.init
