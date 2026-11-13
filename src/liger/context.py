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


from pathlib import Path
import json
from typing import Any, Callable, Self, TypeVar
from liger.config.typing import ConfigLike


T = TypeVar("T", bound=ConfigLike)


class Context:
    """Manages keyed values for use across reinstantiations of contexts."""

    def __init__(self, store: dict[str, Any] = {}):
        self._store = store
        self._used: set[str] = set()

    @classmethod
    def from_frozen(cls, filepath: str | Path) -> Self:
        return cls(json.load(Path(filepath).open()))

    def freeze(self, filepath: str | Path) -> None:
        """Export stored values to a file path."""
        json.dump(self._store, Path(filepath).open("w"), indent=4)

    def once(self, key: str, factory: Callable[[], T]) -> T:
        """Compute a value once, retrieving the stored value on subsequent calls."""
        self._used.add(key)
        if key not in self._store:
            self._store[key] = factory()
        return self._store[key]

    def pin(self, key: str, value: T) -> T:
        """Pin the value of a given key which is computed in each context
        but must remain the same across all contexts.
        """
        self._used.add(key)
        stored = self._store.setdefault(key, value)
        if stored != value:
            raise RuntimeError(f"{key!r} changed value: {stored!r} -> {value!r}")
        return value

    def check_used(self) -> None:
        """Check whether all stored values have been used; raise if not.

        Generally, if values have not been used, a namespace mismatch occurred
        between a previously stored context and this one.
        """
        unused = self._store.keys() - self._used
        if len(unused) > 0:
            raise RuntimeError(
                f"Unused stored values: {unused}; Used values: {self._used}"
            )
