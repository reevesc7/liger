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


from types import FrameType
from typing import Generator
from contextlib import contextmanager
import signal


class SIGTERMReceived(Exception):
    """Raised when a SIGTERM signal is received"""
    pass


def _handle_sigterm(_signalnum: int, _frame: FrameType | None) -> None:
    raise SIGTERMReceived()


@contextmanager
def sigterm_handler() -> Generator[None]:
    old_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old_handler)
