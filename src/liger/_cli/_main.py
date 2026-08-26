# liger - Helper functions for the Likert General Regressor project
# Copyright (C) 2024  Chris Reeves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from argparse import ArgumentParser
from enum import StrEnum
from ._tpot import init_tpot_parser


class _Consumer(StrEnum):
    TPOT = "tpot"


def _make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Start a liger TPOT run")
    subparsers = parser.add_subparsers(dest="consumer", required=True)
    init_tpot_parser(subparsers.add_parser(_Consumer.TPOT, description=(
        "TPOT run commands"
    )))
    return parser


def cmd_liger():
    parser = _make_parser()
    args = parser.parse_args()
    args.func(args)
