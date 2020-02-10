#!/usr/bin/env python3

import os
import subprocess as subp
import sys
import unittest

before_path = sys.path[:]
sys.path.append('../..')
import test_harness as th
sys.path = before_path

def _completion_vars(program, command):
    '''
    Returns a dictionary of bash varibles to trigger bash-completion.

    >>> vals = _completion_vars('foo', 'bar --he')
    >>> sorted(vals.keys())
    ['COMP_CWORD', 'COMP_LINE', 'COMP_POINT', 'COMP_WORDS']
    >>> [vals[x] for x in sorted(vals.keys())]
    [2, 'foo bar --he', 12, 'foo bar --he']

    >>> vals = _completion_vars('foo', 'bar --help ')
    >>> sorted(vals.keys())
    ['COMP_CWORD', 'COMP_LINE', 'COMP_POINT', 'COMP_WORDS']
    >>> [vals[x] for x in sorted(vals.keys())]
    [3, 'foo bar --help ', 15, 'foo bar --help ']
    '''
    line = program + ' ' + command
    words = line.rstrip()
    args = command.split()
    cword = len(args)
    point = len(line)
    if line[-1] == ' ':
        words += ' '
        cword += 1
    return {
        'COMP_LINE': line,
        'COMP_WORDS': words,
        'COMP_CWORD': cword,
        'COMP_POINT': point,
        }

def get_completion(completion_file, program, command):
    '''
    Returns the potential completions from a bash-completion

    @param completion_file (str): file containing bash-completion definition
    @param program (str): program being invoked and completed (e.g., 'cp')
    @param command (str): rest of the command-line args (e.g., '-r ./')
    @return (list(str)): All potential completions if the user were to have
        entered program + ' ' + command into an interactive terminal and hit
        tab twice.
    '''
    replacements = _completion_vars(program, command)
    replacements['completion_file'] = completion_file
    replacements['program'] = program
    full_cmdline = (
        r'source {completion_file};'
        'COMP_LINE="{COMP_LINE}"'
        ' COMP_WORDS=({COMP_WORDS})'
        ' COMP_CWORD={COMP_CWORD}'
        ' COMP_POINT={COMP_POINT};'
        '$(complete -p {program} |'
        ' sed "s/.*-F \\([^ ]*\\) .*/\\1/") &&'
        ' for elem in "${{COMPREPLY[@]}}"; do'
        '   echo "${{elem}}";'
        ' done'.format(**replacements))
    out = subp.check_output(['bash', '-i', '-c', full_cmdline])
    return out.decode('utf-8').splitlines()

class TestFlitBaseCompletion(unittest.TestCase):
    def assertEqualCompletion(self, args, expected):
        'Asserts that the expected completions are obtained'
        completion_file = os.path.join(th.config.bash_completion_dir, 'flit')
        actual = get_completion(completion_file, 'flit', args)
        self.assertEqual(sorted(actual), sorted(expected))

    def test_no_args(self):
        self.assertEqualCompletion('', [
            '-h', '--help',
            '-v', '--version',
            'help',
            'bisect',
            'experimental',
            'import',
            'init',
            'make',
            'update',
            ])

    def test_dash(self):
        self.assertEqualCompletion('-', ['-h', '--help', '-v', '--version'])

    def test_dash_help(self):
        self.assertEqualCompletion('-h', ['-h'])
        self.assertEqualCompletion('-h ', [])
        self.assertEqualCompletion('--help', ['--help'])
        self.assertEqualCompletion('--help ', [])

    def test_dash_version(self):
        self.assertEqualCompletion('-v', ['-v'])
        self.assertEqualCompletion('-v ', [])
        self.assertEqualCompletion('--version', ['--version'])
        self.assertEqualCompletion('--version ', [])

    def test_nonmatching(self):
        self.assertEqualCompletion('-a', [])
        self.assertEqualCompletion('a', [])
        self.assertEqualCompletion('c', [])
        self.assertEqualCompletion('bisecj', [])

    def test_matching_subcommand(self):
        self.assertEqualCompletion('h', ['help'])
        self.assertEqualCompletion('bi', ['bisect'])
        self.assertEqualCompletion('exp', ['experimental'])
        self.assertEqualCompletion('i', ['import', 'init'])
        self.assertEqualCompletion('m', ['make'])
        self.assertEqualCompletion('up', ['update'])

def main():
    'Calls unittest.main(), only printing if the tests failed'
    from io import StringIO
    captured = StringIO()
    old_stderr = sys.stderr
    try:
        sys.stderr = captured
        result = unittest.main(exit=False)
    finally:
        sys.stderr = old_stderr
    if not result.result.wasSuccessful():
        print(captured.getvalue())
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
