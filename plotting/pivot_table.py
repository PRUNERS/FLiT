#!/usr/bin/env python3
'Generates statistics from a given test results csv file'

import csv
import sys
import argparse
import itertools

def parse_args(arguments):
    'Parses the arguments and returns a namedtuple according to argparse'
    parser = argparse.ArgumentParser(
        description='''
            Generates statistics from a given test results csv file.  The csv
            file should have columns "switches", "precision", "sort", "score0",
            "score0d", "host", "compiler", "name", "score1", "score1d".  This
            basically mimics a pivot table functionality.
            ''',
        )
    parser.add_argument('-c', '--columns', default=None,
                        help='''
                            Comma-separated list of table columns to use as the
                            columns.  If empty, then it will just to totals by
                            row.  For example, --columns precision,sort
                            ''')
    parser.add_argument('-r', '--rows', default='name',
                        help='''
                            Comma-separated list of table columns to use as the
                            rows.  For example, --rows name,compiler
                            ''')
    parser.add_argument('-f', '--fix', default=None,
                        help='''
                            Comma-separated list of name=value to remain fixed
                            (i.e.  pre-filter).  For example, --fix
                            precision=d,switches=,compiler=g++
                            ''')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-o', '--output', default='output.csv',
                       help='Output file for generated table counts')
    group.add_argument('-', '--stdout', action='store_true',
                       help='''
                           Write to stdout instead of to a file.  Conflicts
                           with --output.
                           ''')
    parser.add_argument('csvfile')
    return parser.parse_args(args=arguments)


def split_filters(filter_string):
    '''
    Returns a list of 2-tuples of (name, value) in the order given.  The
    filter_string is of the format key1=value1,key2=value2,key3=value3 for as
    many key-value pairs you want.

    You may enclose a value in single or double quotes, but you cannot have a
    comma in a value field since that separates the pairs.
    '''
    if filter_string is None:
        return []

    filters = []
    for fixed_filter in filter_string.split(','):
        filter_split = fixed_filter.split('=')
        assert len(filter_split) == 2, \
                fixed_filter + ' is not a valid --fix parameter'
        filters.append((filter_split[0], filter_split[1].strip('"\'')))
    return filters


def read_csv(filename, filters):
    '''
    Read in the csv applying the filters of only keeping rows that match the
    column -> value pairs in all filters
    '''
    inputrows = []
    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile)
        inputrows = [x for x in reader
                     if all(x[name] == value for name, value in filters)]
    return inputrows


def generate_table(inputrows, col_types, row_types):
    '''
    Creates the table complete with column names and row names

    Returns a tuple of (col_names, row_names, table)
    '''
    row_sets = [set(x[name] for x in inputrows) for name in row_types]
    col_sets = [set(x[name] for x in inputrows) for name in col_types]
    row_names = [x for x in itertools.product(*[sorted(x) for x in row_sets])]
    col_names = [x for x in itertools.product(*[sorted(x) for x in col_sets])]

    assert len(row_names) > 0, \
            'You must select at least one row type using --rows'

    table = []  # list of lists, len(row_names) x len(col_names)
    for row_name in row_names:
        new_table_row = []
        table.append(new_table_row)
        if len(col_names) > 0:
            for col_name in col_names:
                new_table_row.append(len(
                    set(x['score0'] for x in inputrows
                        if all(x[name] == value
                               for name, value in zip(row_types, row_name))
                        and all(x[name] == value
                                for name, value in zip(col_types, col_name)))
                    ))
        else:
            new_table_row.append(len(
                set(x['score0'] for x in inputrows
                    if all(x[name] == value
                           for name, value in zip(row_types, row_name)))
                ))

    if len(col_names[0]) == 0:
        col_names = [('total',)]

    return (col_names, row_names, table)


def write_table_to_csv(outfile, col_names, row_names, table):
    '''
    Writes the table complete with column names, row names, and table contents
    out to the specified output.
    '''
    writer = csv.writer(outfile)

    fill_empty = lambda x: '<empty>' if x == '' else x

    # Write the first row
    if len(col_names[0]) > 1:
        writer.writerow([r'r\c'] + [str(x) for x in col_names])
    else:
        writer.writerow([r'r\c'] + [fill_empty(x[0]) for x in col_names])

    # Write each row
    for name, row in zip(row_names, table):
        if len(name) == 1:
            writer.writerow([fill_empty(name[0])] + row)
        else:
            writer.writerow([str(name)] + row)

def write_table_to_latex(outfile, col_names, row_names, table):
    '''
    Writes the table complete with column names, row names, and table contents
    out to a Latex source file.
    '''
    fill_empty = lambda x: r'$<$empty$>$' if x == '' else x

    if len(col_names[0]) > 1:
        col_names = [str(x) for x in col_names]
    else:
        col_names = [fill_empty(x[0]) for x in col_names]
    col_names = [x.strip() for x in col_names]

    if len(row_names[0]) > 1:
        row_names = [str(x) for x in row_names]
    else:
        row_names = [fill_empty(x[0]) for x in row_names]
    row_names = [x.strip() for x in row_names]
    
    max_value = max([max(x) for x in table])
    bound = lambda lower, val, upper: min(upper, max(lower, val))
    def colorstring(value):
        'Creates a "\color[rgb]{r,g,b}" string for the hard-coded colormap'
        percent = value / max_value
        red = bound(0.5, 0.5 + percent * 5 / 3, 1.0)
        green = bound(0.5, 0.125 + percent * 5 / 4, 1.0)
        blue = bound(0.5, -2/3 + percent * 5 / 3, 1.0)
        return r'\cellcolor[rgb]{' + '{0}, {1}, {2}'.format(red, green, blue) + '}'

    outfile.write(r'''
\documentclass[border=10pt]{standalone}
\renewcommand*{\familydefault}{\sfdefault}
\usepackage{sfmath}
\usepackage{graphicx}
\usepackage{colortbl}
\usepackage{xcolor}

\begin{document}

''')
    outfile.write(r'\begin{tabular}{r' + '|c' * len(col_names) + '}\n')
    outfile.write('  &\n');
    outfile.write('  \\rotatebox{90}{' +
                  '} &\n  \\rotatebox{90}{'.join(col_names) + '}\n')
    outfile.write('  \\\\\n')
    outfile.write('  \\hline\n')
    rows = []
    for i in range(len(row_names)):
        row_string = '  ' + row_names[i] + ' &\n  '
        row_string += ' &\n  '.join(
                [colorstring(x) + ' ' + str(x) for x in table[i]]
                )
        #for elem in table[i][:-1]:
        #    row_string += '  ' + str(elem) + ' &\n'
        #row_string += '  ' + str(table[i][-1]) + '\n'
        rows.append(row_string)
    outfile.write('  \\\\\n  \\hline\n'.join(rows))
    
    outfile.write('\end{tabular}\n')
    outfile.write('\n\end{document}\n')

def main(arguments):
    'Main entry point'
    args = parse_args(arguments)
    filters = split_filters(args.fix)
    inputrows = read_csv(args.csvfile, filters)

    if args.columns is not None:
        col_types = args.columns.split(',')
    else:
        col_types = []
    row_types = args.rows.split(',')

    [col_names, row_names, table] = \
            generate_table(inputrows, col_types, row_types)

    if args.stdout:
        outfile = sys.stdout
    else:
        outfile = open(args.output, 'w')

    try:
        write_table_to_csv(outfile, col_names, row_names, table)
    except:
        if not args.stdout:
            outfile.close()
        raise


if __name__ == '__main__':
    main(sys.argv[1:])

