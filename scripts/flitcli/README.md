# FLiT Architecture

The flit command-line tool is split up modularly into separate subcommands.
Inside of the script folder where flit.py resides, there are a few other
things:

- `README.md`: This documentation
- `flitconfig.py`: Contains paths to FLiT files needed by the command-line tool
- `flit_*.py`: Each subcommand is isolated into its own executable
  python script.  This allows for modularity and easy extensibility.
- `flitelf.py`: helper script for reading ELF binary files
- `flitutil.py`: helper functions used in many subcommands
- `experimental/flit_*.py`: experimental subcommands

## FLiT Subcommands

To add an additional subcommand, there are only a few things you need to do.
For this documentation, let us use the example subcommand of squelch.

1. Create a file in this same directory called `flit_squelch.py`.  The
   subcommand should not be too long in length, otherwise it makes it hard to
   use and the help documentation becomes a bit hard to read.  Other than that,
   there is no technical reason long subcommand names cannot be used.  Try to
   stay within 8 characters.
2. Define a string on the global scope named `brief_description`.  This
   variable is used to generate a brief description of the submodule.  Try to
   keep this to one or two sentences.  More detailed explanations can be given
   by the help documentation of the subcommand.
3. Implement a main method with the following declaration:
   `def main(arguments, prog=None)`.  The `prog` argument is intended to be in
   the help documentation for that subcommand as the executable that was
   called.  It can be passed directly into the `prog` argument for the
   `argparse.ArgumentParser` class.  The `arguments` arg only contain the
   arguments and not the program name (similar to `sys.argv[1:]`).
4. Implement a method  called `populate_parser(parser=None)` where it populates
   a given `argparse.ArgumentParser` instance (if `None` is given, you create
   one in the function) and returns the parser.

## Example Subcommand

`flit_squelch.py`:

```python
'Implements the squelch subcommand'

import argparse
import sys

brief_description = 'Quiets any communication with the server'

def populate_parser(parser=None):
    if not parser:
        parser = ArgumentParser()
    parser.description = '''
        The squelch command rejects any commands to communicate from
        the server.  This is not an actual command or feature of flit,
        this subcommand is only for illustrative purposes on how to
        generate a subcommand.
        '''
    parser.add_argument('-s', '--server', default='127.0.0.1',
                        help='The server IP address')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show verbose messages')
    return parser

def main(arguments, prog=None):
    parser = populate_parser()
    if prog: parser.prog = prog
    args = parser.parse_args(arguments)

    # Subcommand logic here
    # ...

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
```
