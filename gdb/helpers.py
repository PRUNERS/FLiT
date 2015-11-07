#here are the helper functions / classes for geb_extends

import gdb

class qfpWatchpoint (gdb.Breakpoint):
    def stop (self):
        #here we hit and we have to decide whether or not to continue

def watch_handler (event):
    #if 

def execCommands(clist):
    for c in clist:
        gdb.execute(c)

def getPrecString(p):
    if p == 'f':
        return 'float'
    if p == 'd':
        return 'double'
    return 'long double'
