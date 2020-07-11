import subprocess as subp

class Completion():
    def prepare(self, program, command):
        self.program = program
        self.COMP_LINE = program + ' ' + command
        self.COMP_WORDS = self.COMP_LINE.rstrip()

        args=command.split()
        self.COMP_CWORD=len(args)
        self.COMP_POINT=len(self.COMP_LINE)

        if (self.COMP_LINE[-1] == ' '):
            self.COMP_WORDS += " "
            self.COMP_CWORD += 1

    def run(self, completion_file, program, command):
        self.prepare(program, command)
        full_cmdline = \
            r'source {compfile};' \
            'COMP_LINE="{COMP_LINE}"' \
            ' COMP_WORDS=({COMP_WORDS})' \
            ' COMP_CWORD={COMP_CWORD}' \
            ' COMP_POINT={COMP_POINT};' \
            '$(complete -p {program} |' \
            ' sed "s/.*-F \\([^ ]*\\) .*/\\1/") &&' \
            ' for elem in "${{COMPREPLY[@]}}"; do' \
            '   echo "${{elem}}";' \
            ' done'.format(
                compfile=completion_file,
                COMP_LINE=self.COMP_LINE,
                COMP_WORDS=self.COMP_WORDS,
                COMP_POINT=self.COMP_POINT,
                program=self.program,
                COMP_CWORD=self.COMP_CWORD,
                )
        #print(full_cmdline)
        proc = subp.Popen(['bash', '-i', '-c', full_cmdline], stdout=subp.PIPE)
        out, _ = proc.communicate()
        return out.decode('utf-8').splitlines()

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

def main():
    'Compares the two implementations'
    cfile = '../completions/bar'
    prog = 'bar'
    args = 'import '
    complete_1 = Completion().run(cfile, prog, args)
    complete_2 = get_completion(cfile, prog, args)
    assert complete_1 == complete_2
    from pprint import pprint
    pprint(complete_1)

if __name__ == '__main__':
    main()
