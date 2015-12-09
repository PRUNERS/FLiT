# to begin, we will do sanity checks.
# at this point, we should have gdb loaded, this should be pulled
# in after, and we can process.  But let's be sure that we have 
# a valid inferior

import gdb
import os
import sys
sys.path.append(os.getcwd())
from helpers import execCommands, inf_terminated, getDivergence, toggle_inf

#data structures


#this is the event handler for the initial setup -- it
#catches the INT $3 planted in the checkpoint instrumentation
gdb.events.stop.connect(helpers.catch_trap)

execCommands(['run 2> inf1.watch', 'add-inferior -exec inf2',
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

while not inf_terminated(1) and not inf_terminated(2):
    #scan the watchpoints to determine action, either:
    command = ''
    for sub in helpers.subjects:
        if len(sub.watches) > 2:
            print('too many watches detected in subject: ' + sub.label)
            break
        if sub.state == subjState.searching:
            #hit divergence
            if (sub.watches[0].state == watchState.hitCount and
                sub.watches[1].state == watchState.hitCount):
                if not sub.seekDivergence(): #checks for divergence and returns true if we should pursue
            #time to compare data
                    setSearching()
                    continue
                else:
                    sub.setSeeking(div)
            else:
                #the sub is searching, but one or more watches haven't hit count
                #or they've terminated
                curInf = gdb.selected_inferior()
                otherInf = sub.getOtherInf()
                curWatch = sub.getWatch(curInf)
                if inf_terminated(curInf):
                    if (sub.getWatch(otherInf).state == watchState.searching or
                        inf_terminated(otherInf):
                        sub.toggle_inf()
                        gdb.execute('continue')
                    #what else could be the case now?
                    
                if (curWatch.state == watchState.hitSeek and
                    sub.watches[sub.].state == watchState.searching):
                    sub.toggle_inf()
                    gdb.execute('continue')
            
                
#isolated a divergence

#normal termination
