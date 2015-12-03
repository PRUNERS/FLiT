# to begin, we will do sanity checks.
# at this point, we should have gdb loaded, this should be pulled
# in after, and we can process.  But let's be sure that we have 
# a valid inferior

import gdb
import os
import sys
sys.path.append(os.getcwd())
import helpers

#data structures



# if len(gdb.inferiors()) != 1 || !gdb.selected_inferior().is_valid():
#     print('in ' + __file__ + ' should have one valid inferior, found 0, quitting . . .')
#     gdb.execute('quit')

# some data structures. 
## get needed environment info
# precision = getPrecString(os.environ.get("PREC"))
# test = os.environ.get("TEST")
##this is our list of critical data, one list per test
# criticals = [['Globals<' + precision + '>::sum', '*(' + precision + '*)' +
#               lookupAddress('Globals<' + precision + '>::prods[0]']),
#               ['Globals<' + precision + '>::sum', '*(' + precision + '*)' +
#                lookupAddress('Globals<' + precision + '>::prods[0]')]]
              
# the initialization -- load other inferior and setup rewind (i.e. record)

gdb.events.stop.connect(helpers.catch_trap)

helpers.execCommands(['run 2> inf1.watch', 'add-inferior -exec inf2',
                      'inferior 2', 'run 2> inf2.watch'])
              
# init_commands = ['add-inferior -exec inf2', 'run', 'inferior 2', 'run', 'record']

# execCommands(init_commands)

# #next we set up the watchpoints
# ## here we need to use the python extensions for our
# ## custom qfpWatchpoint and registering callbacks 
# execCommands(['inferior 1'])
# for c in criticals[test]:
#     qfpWatchpoint.__init__(c)
# execCommands(['inferior 2'])
# for c in criticals[test]:
#     qfpWatchpoint.__init__(c)
                 
# #we should be ready to go now, and let the watchpoints do their work

# execCommands(['run', 'inferior 1', 'run'])

