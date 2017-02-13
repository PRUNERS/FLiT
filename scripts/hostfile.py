#this python file contains the DB_HOST and RUN_HOSTS objects,
#along with COMPILERS

#DB_HOST: (user, fqdn)

DB_HOST = ('sawaya', 'bihexal.cs.utah.edu')


#the format of RUN_HOSTS tuples: (user, fqdn, procs, host_script,
#   [CUDA only: (True | False (default))],
#   [Collect opcodes: (True | False (default))

RUN_HOSTS = (('u0422778', 'kingspeak.chpc.utah.edu', 56, 'kingspeak_cpu_collect',
              False, True),
             ('u0422778', 'kingspeak.chpc.utah.edu', 1, 'kingspeak_gpu_collect',
              True, False),
             ('sawaya', 'ms0345.utah.cloudlab.us', 8, None,
              False, False))

 
#these tuples are so formated: (COMPILER [CLANG, GCC, INTEL, NVCC], 5.4)

COMPILERS = (('CLANG', '3.9'),
             ('GCC', '5.4'),
             ('INTEL', '16.0.3'),
             ('NVCC', '7.5'))

