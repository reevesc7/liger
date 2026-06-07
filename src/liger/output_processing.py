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


from typing import Any, Iterable, Iterator, overload
from pathlib import Path
import json


class JSONCache:
    def __init__(
        self,
        paths: Path | str | Iterable[Path | str],
        patterns: str | Iterable[str],
    ) -> None:
        if isinstance(paths, (Path, str)):
            paths = paths,
        if isinstance(patterns, str):
            patterns = patterns,
        filepaths: set[Path] = set()
        for path in paths:
            path = Path(path)
            if not path.exists():
                print(f"Warning: {path} does not exist")
                continue
            filepaths.update(self._filepaths_matching_patterns(path, patterns))
        self.filepaths = sorted(filepaths)
        self._cache: dict[Path, Any] = {}

    @staticmethod
    def _filepaths_matching_patterns(path: Path, patterns: Iterable[str]) -> set[Path]:
        if path.is_file() and any(path.match(pattern) for pattern in patterns):
            return {path}
        return {
            fpath
            for pattern in patterns
            for fpath in sorted(path.rglob(pattern))
            if fpath.is_file()
        }

    def get_file_data(self, filepath: Path | str) -> Any:
        filepath = Path(filepath)
        if filepath not in self._cache:
            if filepath not in self.filepaths:
                self.filepaths.append(filepath)
            with open(filepath, "r") as file:
                self._cache[filepath] = json.load(file)
        return self._cache[filepath]

    def __iter__(self) -> Iterator[tuple[Path, Any]]:
        for filepath in self.filepaths:
            yield filepath, self.get_file_data(filepath)


def _json_load(filepath: Path) -> Any:
    with open(filepath, "r") as file:
        return json.load(file)


def mass_json_load(
    paths: Path | str | Iterable[Path | str],
    pattern: str,
) -> Iterator[Any]:
    if isinstance(paths, (Path, str)):
        paths = paths,
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"Warning: {path} does not exist")
            continue
        if path.is_file() and path.match(pattern) and path.suffix == ".json":
            yield _json_load(path)
        for fpath in sorted(path.rglob(pattern)):
            if not fpath.suffix == ".json":
                continue
            yield _json_load(fpath)


_sentinel = object()

@overload
def mass_dict_get(dicts: Iterable[dict[str, Any]], key: str) -> list[Any]: ...
@overload
def mass_dict_get(dicts: Iterable[dict[str, Any]], key: str, default: Any) -> list[Any]: ...

def mass_dict_get(dicts: Iterable[dict[str, Any]], key: str, default: Any = _sentinel) -> list[Any]:
    if default is _sentinel:
        return [dictionary.get(key) for dictionary in dicts]
    return [dictionary.get(key, default) for dictionary in dicts]

