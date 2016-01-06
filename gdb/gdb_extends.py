#!/usr/bin/env python3

# to begin, we will do sanity checks.
# at this point, we should have gdb loaded, this should be pulled
# in after, and we can process.  But let's be sure that we have 
# a valid inferior

import gdb
import os
import sys
sys.path.append(os.getcwd())
import helpers
from enum import Enum

#data structures


#this is the event handler for the initial setup -- it
#catches the INT $3 planted in the checkpoint instrumentation
gdb.events.stop.connect(helpers.catch_trap)

#this event handler notifies us when an inferior exits
gdb.events.exited.connect(helpers.catch_term)

helpers.execCommands(['run 2> inf1.watch', 'add-inferior -exec inf2',
                       'inferior 2', 'run 2> inf2.watch'])

#at this point, either:
## both inferiors have terminated
## we've just added a watchpoint in each inferior (hopefully the same logical location)


              
#this (next) loop will not exit until either:
## we've exhausted the execution of both inferiors.


## and no divergence has been found OR
## 
## we've identified a (first) divergence location and
## the gdb context for each inf as at the point of divergence
## (immediately before writing the divergent results to the
## watch points

# execution will return here when:
## 1. the hit count reaches CPERIOD0
## 2. an inferior terminates
## 3. the inferior is located at a divergence (i.e. count == target)
## 4. a signal was caught

#this should be the beginning

#here are our exec states
class execState(Enum):
    init = 1
    search1 = 2
    search2 = 3
    analyze = 4
    seek1 = 5
    seek2 = 6
    user = 7

#sub = helpers.subjects[0]
estate = execState.init

#print('subject state is: ' + str(sub.state))

#res = input('would you like to return to user control? [y|n]')
res = 'n'
for lab, sub in helpers.subjects.items():
    if res.lower() == 'n' and sub.state == helpers.subjState.searching:
        sub.toggle_inf()
        estate = execState.search1
        inf1 = gdb.selected_inferior().num
        watch1 = sub.getWatch(inf1)
        inf2 = sub.getOtherInf()
        watch2 = sub.getWatch(inf2)
        curInf = inf1
        curW = watch1
        #we're ready to search
        print('inf1 is: ' + str(inf1) + ' and inf2 is: ' + str(inf2))
        print('watch1 inf is: ' + str(watch1.inf) + ' and watch2 inf is: ' +
              str(watch2.inf))
        while True:
            if estate == execState.search1:
                print('handling search1 state')
                print('inf1 masterCount is: ' + str(watch1.masterCount))
                helpers.execCommands(['continue'])
                if (watch1.state == helpers.watchState.hitCount or
                    watch1.state == helpers.watchState.infExited):
                    estate = execState.search2
                    sub.toggle_inf()  #activates the other inferior
                else:
                    gdb.error('reached unknown state after execState.search1')
            if estate == execState.search2:
                print('handling search2 state')
                print('inf2 masterCount is: ' + str(watch2.masterCount))
                helpers.execCommands(['continue'])
                if (watch2.state == helpers.watchState.hitCount or
                    watch2.state == helpers.watchState.infExited):
                    estate = execState.analyze
                else:
                    gdb.error('reached unknown state after execState.search2')
                    break
            if estate == execState.analyze:
                print('handling analyze state')
                print('inf1 masterCount is: ' + str(watch1.masterCount))
                print('inf2 masterCount is: ' + str(watch2.masterCount))
                div = sub.seekDivergence()
                print('hit analyze state with div = ' + str(div))
                if div != -1: #means that the user chose to focus on identified div
                    gdb.events.exited.disconnect(helpers.catch_term)
                    sub.setSeeking(div)
                    gdb.events.exited.connect(helpers.catch_term)
                    watch1 = sub.watches[0]
                    watch2 = sub.watches[1]
                    inf1 = watch1.inf
                    inf2 = watch2.inf
                    estate = execState.seek1
                else:
                    if (watch1.state == helpers.watchState.infExited and
                        watch2.state == helpers.watchState.infExited):
                        print('hit infExited in main control loop')
                        print('no divergence detected in specified regions')
                        estate = execState.user
                        #helpers.execCommands(['quit'])
                    else:
                        sub.setSearching()
                        sub.toggle_inf()
                        estate = execState.search1
                    if (watch1.state == helpers.watchState.infExited or
                        watch2.state == helpers.watchState.infExited):
                        gdb.error('something wrong -- after analyze one terminated inferior')
                        break
                #sub.toggle_inf()
            if estate == execState.seek1:
                print('handling seek1 state')
                #break
                # #DELME
                # print('returning control at seek1')
                # break
                # ##
                # TODO -- testing 'record'
                #helpers.execCommands(['record', 'continue'])
                helpers.execCommands(['continue'])
                #DELME
                print('after continue, watch1.state and watch2.state are: ' +
                      str(watch1.state) + ' and ' + str(watch2.state))
                #
                if watch1.state == helpers.watchState.hitSeek:
                    estate = execState.seek2
                    #TODO testing record on inf1
 #                   helpers.execCommands(['record stop', 'record save inf1.rec'])
                    sub.toggle_inf()
                else:
                    gdb.error('couldn\'t reach target in seek of inf ' + str(inf1))
                    break
            if estate == execState.seek2:
                print('handling seek2 state')
                #break
                helpers.execCommands(['continue'])
                if watch2.state == helpers.watchState.hitSeek:
                    estate = execState.user
                else:
                    gdb.error('couldn\'t reach target in seek of inf ' + str(inf2))
                    break
            if estate == execState.user:
                break
        
    
    

# while not inf_terminated(1) and not inf_terminated(2):
#     #scan the watchpoints to determine action, either:
#     command = ''
#     for sub in helpers.subjects:
#         if len(sub.watches) > 2:
#             print('too many watches detected in subject: ' + sub.label)
#             break
#         if sub.state == subjState.searching:
#             #hit divergence
#             if (sub.watches[0].state == watchState.hitCount and
#                 sub.watches[1].state == watchState.hitCount):
#                 if not sub.seekDivergence(): #checks for divergence and returns true if we should pursue
#             #time to compare data
#                     setSearching()
#                     continue
#                 else:
#                     sub.setSeeking(div)
#             else:
#                 #the sub is searching, but one or more watches haven't hit count
#                 #or they've terminated
#                 curInf = gdb.selected_inferior()
#                 otherInf = sub.getOtherInf()
#                 curWatch = sub.getWatch(curInf)
#                 if inf_terminated(curInf):
#                     if (sub.getWatch(otherInf).state == watchState.searching or
#                         inf_terminated(otherInf):
#                         sub.toggle_inf()
#                         execCommands(['continue'])
#                     #what else could be the case now?
                    
#                 if (curWatch.state == watchState.hitSeek and
#                     sub.watches[sub.].state == watchState.searching):
#                     sub.toggle_inf()
#                     execCommands(['continue'])
            
                
#isolated a divergence

#normal termination
