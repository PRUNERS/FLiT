# to begin, we will do sanity checks.
# at this point, we should have gdb loaded, this should be pulled
# in after, and we can process.  But let's be sure that we have 
# a valid inferior

import gdb
import os
import sys
sys.path.append(os.getcwd())
from helpers import execCommands, inf_terminated, getDivergence,

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

while not inf_terminated(1) and not inf_terminated(2)
    #scan the watchpoints to determine action, either:
    command = ''
    for (addr, watches) in helpers.wplist:
        if len(watches) > 2:
            print('non-symmetry detected in memory accesses between inferiors.')
            break
        if watches[0].count == CPERIOD and watches[1].count == CPERIOD:
            #time to compare data
            div = getDivergence(addr)
            if div > -1:
                watches[0].target = watches[1].target = div
                command = ['infer 1', 'run'
            for w in watches:
                with w:
                    masterCount = count + masterCount
                    count = 0
            else:
                for w in watches:
                    with w:
                        target = -1
                        masterCount = count + masterCount
                        
                
#isolated a divergence

#normal termination
