import abc
import argparse
import unittest
import sys

def is_option_action(action):
    '''
    If the ArgumentParser action (which is ArgumentParser's way to store
    possible arguments) is an option argument (meaning it starts with a "-"),
    then return True.

    >>> p = argparse.ArgumentParser()
    >>> is_option_action(p.add_argument('--number'))
    True

    >>> is_option_action(p.add_argument('another_number'))
    False

    >>> subparsers = p.add_subparsers()
    >>> _ = subparsers.add_parser('subcommand')
    >>> is_option_action(subparsers)
    False
    '''
    return bool(action.option_strings)

def is_position_action(action):
    '''
    If the ArgumentParser action (which is ArgumentParser's way to store
    possible arguments) is a positional argument, then return True.

    >>> p = argparse.ArgumentParser()
    >>> is_position_action(p.add_argument('--number'))
    False

    >>> is_position_action(p.add_argument('another_number'))
    True

    >>> subparsers = p.add_subparsers()
    >>> _ = subparsers.add_parser('subcommand')
    >>> is_position_action(subparsers)
    True
    '''
    return not is_option_action(action)

def is_subparser_action(action):
    '''
    If the ArgumentParser action (which is ArgumentParser's way to store
    possible arguments) is a subcommand with its own argument parsing, then
    return True.

    Note: all subparser actions are also position actions.

    >>> p = argparse.ArgumentParser()
    >>> is_subparser_action(p.add_argument('--number'))
    False

    >>> is_subparser_action(p.add_argument('another_number'))
    False

    >>> subparsers = p.add_subparsers()
    >>> _ = subparsers.add_parser('subcommand')
    >>> is_subparser_action(subparsers)
    True
    '''
    return isinstance(action, argparse._SubParsersAction)

_p = argparse.ArgumentParser()
_action_map = _p._registries['action']
_name_map = {v: k for k, v in _action_map.items()}
del _p
del _action_map

class ActionInspector:
    '''
    All Actions have:
    - option_strings
    - dest
    - nargs
    - const
    - default
    - type
    - choices
    - required
    - help
    - metavar

    argparse._VersionAction adds:
    - version

    argparse._SubParsersAction adds:
    - prog
    - parser_class
    '''

    NAME_MAP = _name_map

    def __init__(self, action):

        if action.__class__ in self.NAME_MAP:
            self.action_type = self.NAME_MAP[action.__class__]
        else:
            self.action_type = action.__class__.__name__
            print('Warning: unrecognized action class: {}'.format(self.action_type),
                  file=sys.stderr)

        # passthrough of attributes
        self.action = action
        self.option_strings = action.option_strings
        self.dest = action.dest
        self.nargs = action.nargs
        self.const = action.const
        self.default = action.default
        self.type = action.type
        self.choices = action.choices
        if is_subparser_action(action):
            self.choices = {key: ParserInspector(val)
                            for key, val in action.choices.items()}
        self.required = action.required
        self.help = action.help
        self.metavar = action.metavar

        # optional attributes
        for attr in ('version', 'prog', 'parser_class'):
            if hasattr(action, attr):
                setattr(self, attr, getattr(action, attr))
            else:
                setattr(self, attr, None)

class ParserInspector:
    'Introspection on an argparse.ArgumentParser'

    def __init__(self, parser):
        self.parser = parser
        self.option_actions = [
            ActionInspector(a) for a in self.parser._optionals._group_actions]
        self.position_actions = [
            ActionInspector(a) for a in self.parser._positionals._group_actions
            if not is_subparser_action(a)]
        self.subparser_actions = []
        if self.parser._subparsers:
            self.subparser_actions = [
                ActionInspector(a) for a in self.parser._subparsers._group_actions
                if is_subparser_action(a)]
        self.actions = self.option_actions + self.position_actions + \
                       self.subparser_actions

        self.option_strings = sum(
            [a.option_strings for a in self.option_actions], [])
        self.subparser_choices = sum(
            [list(a.choices) for a in self.subparser_actions], [])

class ArgParseTestBase(unittest.TestCase, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def bashcomplete(self, args):
        '''
        Return a list of completions from the current string of arguments.

        This is an abstract method that must be implemented by derivative
        classes.
        '''
        pass

    def assertEqualCompletion(self, args, expected_completions, msg=None):
        'Asserts that the expected completions are obtained'
        actual = self.bashcomplete(args)
        self.assertEqual(set(expected_completions), set(actual), msg=msg)

    def assertCompletionContains(self, args, expected_subset, msg=None):
        'Asserts that the expected completions are found in the actual'
        actual = self.bashcomplete(args)
        self.assertLessEqual(set(expected_subset), set(actual), msg=msg)

    def assertEmptyAvailableOptions(self, parser, cli=''):
        '''
        Asserts that all options and subparser choices are present in the bash
        completion for the given parser and the current cli.

        Note: The cli is only given to specify which subparser we are currently
        doing.  The parser given should match the subparser associated with the
        cli string.
        '''
        inspector = ParserInspector(parser)

        expected_completions = inspector.option_strings
        if not inspector.position_actions:
            expected_completions.extend(inspector.subparser_choices)

        self.assertCompletionContains(
            cli, expected_completions,
            msg='args: {}'.format(repr(cli)))

    def assertEachSubparserEmptyAvailableOptions(self, parser, cli=''):
        '''
        Asserts that all options and subparser choices are present for the
        current parser and for all subparsers recursively.  This method calls
        assertEmptyAvailableOptions() for the given parser and for all
        subparsers recursively.
        '''
        inspector = ParserInspector(parser)

        # test this parser level
        self.assertEmptyAvailableOptions(parser, cli)

        # test all subparsers recursively
        for choice in inspector.subparser_choices:
            subinspector = inspector.subparser_actions[0].choices[choice]
            self.assertEachSubparserEmptyAvailableOptions(
                subinspector.parser, cli='{} {} '.format(cli, choice))
