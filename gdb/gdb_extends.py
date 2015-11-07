# to begin, we will do sanity checks.
# at this point, we should have gdb loaded, this should be pulled
# in after, and we can process.  But let's be sure that we have 
# a valid inferior

import gdb
from os import environ
import helpers

if len(gdb.inferiors()) != 1 || !gdb.selected_inferior().is_valid():
    print('in ' + __file__ + ' should have one valid inferior, found 0, quitting . . .')
    gdb.execute('quit')

# some data structures. 
## get needed environment info
precision = getPrecString(os.environ.get("PREC"))
test = os.environ.get("TEST")
##this is our list of critical data, one list per test
criticals = [['Globals<' + precision + '>::sum', '*(' + precision + '*)' +
              lookupAddress('Globals<' + precision + '>::prods[0]']),
              ['Globals<' + precision + '>::sum', '*(' + precision + '*)' +
               lookupAddress('Globals<' + precision + '>::prods[0]')]]
              
# the initialization -- load other inferior and setup rewind (i.e. record)
              
init_commands = ['add-inferior -exec file2', 'break main', 'run', 'inferior 2', 'run', 'record', 'delete 1']

execCommands(init_commands)

#next we set up the watchpoints
## here we need to use the python extensions for our
## custom qfpWatchpoint and registering callbacks 
execCommands(['inferior 1'])
for c in criticals[test]:
    qfpWatchpoint.__init__(c)
execCommands(['inferior 2'])
for c in criticals[test]:
    qfpWatchpoint.__init__(c)
                 
#we should be ready to go now, and let the watchpoints do their work

execCommands(['run', 'inferior 1', 'run'])

