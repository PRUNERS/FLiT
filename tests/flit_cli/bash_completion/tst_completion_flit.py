#!/usr/bin/env python3

import os
import sys
import unittest

import util.arginspect as arginspect
from util.completion import get_completion

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
prev_path = sys.path
sys.path.append(os.path.join(SCRIPT_DIR, '..', '..'))
import test_harness as th
sys.path = prev_path

class TestArgparse_Flit(arginspect.ArgParseTestBase):
    FLIT_PROG = 'flit'
    FLIT_BASH_COMPLETION = os.path.join(
        th._flit_dir, 'scripts', 'bash-completion', 'flit')

    def bashcomplete(self, args):
        return get_completion(self.FLIT_BASH_COMPLETION, self.FLIT_PROG, args)

    def get_parser(self):
        subcommands = th.flit.load_subcommands(th.config.script_dir)
        subcommands.append(th.flit.create_help_subcommand(subcommands))
        return th.flit.populate_parser(subcommands=subcommands)

    def test_empty_available_options(self):
        self.assertEmptyAvailableOptions(self.get_parser())

    def test_empty_available_options_for_subparsers(self):
        self.assertEachSubparserEmptyAvailableOptions(self.get_parser())

    def test_no_positional_args(self):
        inspector = arginspect.ParserInspector(self.get_parser())
        # test that there are no positional arguments
        self.assertEqual(inspector.position_actions, [])

if __name__ == '__main__':
    sys.exit(th.unittest_main())
