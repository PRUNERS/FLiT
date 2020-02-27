from io import StringIO
import os
import sys

before_path = sys.path[:]
sys.path.append('../..')
import test_harness as th
sys.path = before_path

class BisectTestError(RuntimeError): pass

def flit_init(directory):
    'Creates a new FLiT directory returning the standard output'
    with StringIO() as ostream:
        retval = th.flit.main(['init', '-C', directory], outstream=ostream)
        if retval != 0:
            raise BisectTestError(
                'Could not initialize (retval={0}):\n'.format(retval) +
                ostream.getvalue())
        init_out = ostream.getvalue().splitlines()
    return init_out

def flit_update(directory):
    'Calls flit update in the given directory'
    with StringIO() as ostream:
        retval = th.flit.main(['update', '-C', directory],
                              outstream=ostream)
        if retval != 0:
            raise BisectTestError('Could not update Makefile\n' +
                                  ostream.getvalue())

def bisect_compile(compiler, directory=None, linker=None, ldflags=None,
                   add_ldflags=None):
    '''
    Runs bisect with the given compiler and returns makefile variables

    @param compiler (str): name or path to invoke the compiler under test
    @param directory (str): directory path to flit test directory
        If None is given, use the current working directory
    @param linker (str): Linker program to use
        If None is given, use baseline compiler
    @param ldflags (str): Linker flags to use
        If None is given, use value from flit-config.toml for the linker
        executable (if given)
    @param add_ldflags (str): Linker flags to add to the base linker flags
        (either from the ldflags given or the ones from the baseline compiler)
        If None is given, no linker flags are added

    @return Dictionary of Makefile variables for first bisect Makefile
    '''

    # create static variable i for how many times this function is called
    if not hasattr(bisect_compile, 'i'):
        bisect_compile.i = 0  # initialize only once
    bisect_compile.i += 1

    # Note: we give a bad flag to cause bisect to error out early
    #   we just need it to get far enough to create the first makefile
    args = ['bisect']

    if directory is not None:
        args.extend(['-C', directory])
    if linker is not None:
        args.extend(['--use-linker', linker])
    if ldflags is not None:
        args.extend(['--ldflags={}'.format(ldflags)])
    if add_ldflags is not None:
        args.extend(['--add-ldflags={}'.format(add_ldflags)])

    args.extend([
        '--precision', 'double',
        compiler + ' -bad-flag',
        'EmptyTest'
        ])

    with StringIO() as ostream:
        retval = th.flit.main(args, outstream=ostream)
        if retval == 0:
            raise BisectTestError('Expected bisect to fail\n' +
                                  ostream.getvalue())

    # Since each call creates a separate bisect dir, we use our counter
    makepath = os.path.join(directory,
                            'bisect-{0:02d}'.format(bisect_compile.i),
                            'bisect-make-01.mk')
    makevars = th.util.extract_make_vars(
        makefile=os.path.relpath(makepath, directory),
        directory=directory)
    return makevars

