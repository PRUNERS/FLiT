#this python file contains the DB_HOST and RUN_HOSTS objects

#DB_HOST: (user, fqdn)
DB_HOST = ('sawaya', 'bihexal.cs.utah.edu')

#the format of RUN_HOSTS tuples: (user, fqdn, procs, [CUDA only: (True | False)])
RUN_HOSTS = (('u0422778', 'kingspeak.chpc.utah.edu', 56, False,
              kingspeak_cpu_startup),
             ('u0422778', 'kingspeak.chpc.utah.edu', 1, True,
              kingspeak_gpu_startup),
             ('sawaya', 'ms0345.utah.cloudlab.us', 8, False, ''))

 
