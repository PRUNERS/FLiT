#this is where the user configures her database and
#worker hosts.
#db_host: (user, fqdn)
#run_host: (user, fqdn, processes, SlurmScript, CUDA only, get opcode count)

import os
import multiprocessing

user = os.environ['USER']
cores = multiprocessing.cpu_count()

DB_HOST = (user, 'localhost')
RUN_HOSTS = ((user, 'localhost', cores, None, False, False),)

#another possibility:

#RUN_HOSTS = ((user, 'localhost', cores, kingspeak_gpu_startup, True, False),)
