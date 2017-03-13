#this is where the user configures her database and
#worker hosts.
#db_host: (user, fqdn)
#run_host: (user, fqdn, processes, SlurmScript, CUDA only, get opcode count)

import os
import multiprocessing

user = os.environ['USER']
cores = multiprocessing.cpu_count()

DB_HOST = (user, 'localhost') 
#DB_HOST = ('sawaya', 'bihexal.cs.utah.edu') 
# RUN_HOSTS = (('u0422778', 'kingspeak1.chpc.utah.edu', 56,
#               'kingspeak_cpu_startup', False, False),
#              ('u0422778', 'kingspeak1.chpc.utah.edu', 12,
#               'kingspeak_gpu_startup', True, False),
#              ('sawaya', 'ms0138.utah.cloudlab.us', 8,
#               None, False, False),)

#another possibility:
RUN_HOSTS = ((user, 'localhost', cores, None, False, False),)
