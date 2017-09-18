'Implements the make subcommand'

import argparse
import multiprocessing
import subprocess
import sys

import flit_import

brief_description = 'Runs the make and adds to the database'

def main(arguments, prog=sys.argv[0]):
    parser = argparse.ArgumentParser(
            prog=prog,
            description='''
                This command runs the full set of tests and adds the results
                to the configured database.
                ''',
            )
    processors = multiprocessing.cpu_count()
    parser.add_argument('-j', '--jobs', type=int, default=processors,
                         help='''
                             The number of parallel jobs to use for the call to
                             GNU make when performing the compilation.  Note,
                             this is not used when executing the tests.  This
                             is because in order to get accurate timing data,
                             one cannot in general run multiple versions of the
                             same code in parallel.
                             ''')
    parser.add_argument('--exec-jobs', type=int, default=1,
                        help='''
                            The number of parallel jobs to use for the call to
                            CNU make when performing the test executtion after
                            the full compilation has finished.  The default is
                            to only run one test at a time in order to allow
                            them to not conflict and to generate accurate
                            timing measurements.
                            ''')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress the Makefile output')
    parser.add_argument('--make-args',
                        help='Arguments passed to GNU Make, separated by commas')
    parser.add_argument('-l', '--label', default='Automatically generated label',
                        help='''
                            The label to attach to the run.  Only applicable
                            when creating a new run.  This argument is ignored
                            if --append is specified.  The default label is
                            ''')
    args = parser.parse_args(arguments)

    check_call_kwargs = dict()
    if args.quiet:
        check_call_kwargs['stdout'] = subprocess.DEVNULL
        #check_call_kwargs['stderr'] = subprocess.DEVNULL
    make_args = []
    if args.make_args is not None:
        make_args = args.make_args.split(',')

    # TODO: can we make a progress bar here?
    print('Calling GNU Make for the runbuild')
    subprocess.check_call([
            'make',
            'runbuild',
            '-j{0}'.format(args.jobs),
        ] + make_args,
        **check_call_kwargs
        )
    print('Calling GNU Make to execute the tests')
    subprocess.check_call([
            'make',
            'run',
            '-j{0}'.format(args.exec_jobs),
        ] + make_args,
        **check_call_kwargs
        )
    print('Importing into the database')
    status = flit_import.main(['--label', args.label])
    if status != 0:
        return status

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
